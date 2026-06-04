# The goal here is to clearly define the scheduler that will be improved over time.


import pandas as pd
from ortools.sat.python import cp_model
from .model import APICModel, PlannedSegment


class APICScheduler:
    def __init__(self, model: APICModel):
        self.model = model

    def transfer_cost(self, dev_a, dev_b, mb):
        if dev_a == dev_b:
            return 0
        if not self.model.links.has_edge(dev_a, dev_b):
            return 10**8
        best = None
        for _, data in self.model.links[dev_a][dev_b].items():
            c = int(round(1000 * mb / max(data["bandwidth"], 1e-9) + 1000 * data["latency"]))
            best = c if best is None else min(best, c)
        return best if best is not None else 10**8

    def migration_cost(self, src_device, dst_device, task):
        if src_device == dst_device:
            return 0.0
        base = self.transfer_cost(src_device, dst_device, task.data_in + task.data_out)
        locality = 500.0 if task.current_device and task.current_device != dst_device else 0.0
        return float(base + locality)

    def best_device_for(self, task):
        best_dev = None
        best_score = None
        for d, dev in self.model.devices.items():
            base = 1000 * task.compute / max(dev.speed, 1e-9)
            penalty = 0 if not task.preferred_kind or task.preferred_kind == dev.kind else 3000
            score = base + penalty
            if best_score is None or score < best_score:
                best_score = score
                best_dev = d
        return best_dev

    def remaining_compute(self, task_name):
        task = self.model.tasks[task_name]
        return max(0, task.compute - task.progress)

    def candidate_segments(self, task_name, now, max_segments=4):
        task = self.model.tasks[task_name]
        remaining = self.remaining_compute(task_name)
        if remaining <= 0:
            return []

        candidates = []
        for d, dev in self.model.devices.items():
            seg_compute = min(remaining, max(1, int(dev.speed * task.segment_ms / 1000)))
            duration = max(1, int(round(1000 * seg_compute / max(dev.speed, 1e-9))))
            candidates.append(PlannedSegment(
                task=task.name,
                segment_id=task.progress // max(1, seg_compute),
                device=d,
                start=now,
                end=now + duration,
                duration=duration,
                compute=seg_compute,
                migrated=(task.current_device != "" and task.current_device != d),
                resumed=(task.progress > 0),
            ))
        candidates.sort(key=lambda s: s.duration)
        return candidates[:max_segments]

    def score_segment(self, seg):
        task = self.model.tasks[seg.task]
        dev = self.model.devices[seg.device]
        duration_cost = seg.duration
        pref_penalty = 0 if not task.preferred_kind or task.preferred_kind == dev.kind else 3000
        mig_penalty = self.migration_cost(task.current_device or seg.device, seg.device, task) if seg.migrated else 0
        late_penalty = 10000 if seg.end > task.deadline else 0
        return duration_cost + pref_penalty + mig_penalty + late_penalty

    def choose_next_segment(self, task_name, now):
        candidates = self.candidate_segments(task_name, now)
        if not candidates:
            return None
        return min(candidates, key=self.score_segment)

    def apply_planned_segment(self, seg):
        task = self.model.tasks[seg.task]
        task.progress = min(task.compute, task.progress + seg.compute)
        task.current_device = seg.device

    def plan_window_segments(self, now):
        planned = []
        for task_name, task in self.model.tasks.items():
            if task.progress >= task.compute:
                continue
            seg = self.choose_next_segment(task_name, now)
            if seg is not None:
                planned.append(seg)
        planned.sort(key=lambda s: (s.end, s.duration, s.task))
        return planned

    def solve_window(self, horizon=30000, fixed_starts=None, fixed_device=None, hints=None):
        fixed_starts = fixed_starts or {}
        fixed_device = fixed_device or {}
        hints = hints or {}

        # Definition of strategies...
        # --- 1) Try with complete constraints (deadline) ---
        df = self._solve_cp(horizon, fixed_starts, fixed_device, hints, relax_deadline=False)
        if df is not None:
            return df

        #print("WARNING: CP-SAT infeasible with deadlines, trying relaxed deadlines")

        # --- 2) Try by relaxing the deadline constraint ---
        df = self._solve_cp(horizon, fixed_starts, fixed_device, hints, relax_deadline=True)
        if df is not None:
            return df

        print("WARNING: CP-SAT still infeasible, falling back to heuristic plan")

        # --- 3) Simple heuristic: sequential planning ---
        return self._heuristic_plan(horizon)
        
        
    def _solve_cp(self, horizon, fixed_starts, fixed_device, hints, relax_deadline: bool):
        model = cp_model.CpModel()
        T = list(self.model.tasks)
        D = list(self.model.devices)

        x, s, e, dur = {}, {}, {}, {}
        intervals_by_dev = {d: [] for d in D}

        for t in T:
            task = self.model.tasks[t]
            s[t] = model.NewIntVar(0, horizon, f"s_{t}")
            e[t] = model.NewIntVar(0, horizon, f"e_{t}")
            dur[t] = model.NewIntVar(1, horizon, f"d_{t}")

            if t in fixed_starts:
                model.Add(s[t] == fixed_starts[t])

            for d in D:
                x[(t, d)] = model.NewBoolVar(f"x_{t}_{d}")
            model.Add(sum(x[(t, d)] for d in D) == 1)

            if t in fixed_device:
                for d in D:
                    model.Add(x[(t, d)] == (1 if d == fixed_device[t] else 0))

            for d in D:
                dev = self.model.devices[d]
                base = max(1, int(round(1000 * task.compute / max(dev.speed, 1e-9))))
                pen = 0 if not task.preferred_kind or task.preferred_kind == dev.kind else 3000
                model.Add(dur[t] == base + pen).OnlyEnforceIf(x[(t, d)])

            model.Add(e[t] == s[t] + dur[t])

            # Strict or relaxed deadline
            if not relax_deadline:
                model.Add(e[t] <= task.deadline)
            else:
                # We allow e[t] > deadline, the penalty will be in the objective.
                pass

            for d in D:
                intervals_by_dev[d].append(
                    model.NewOptionalIntervalVar(
                        s[t], dur[t], e[t], x[(t, d)], f"int_{t}_{d}"
                    )
                )

        for d in D:
            model.AddNoOverlap(intervals_by_dev[d])
            model.Add(
                sum(self.model.tasks[t].memory * x[(t, d)] for t in T)
                <= self.model.devices[d].memories[0].capacity
            )
            model.Add(sum(x[(t, d)] for t in T) <= self.model.devices[d].threads)

        for dep in self.model.deps:
            for da in D:
                for db in D:
                    b = model.NewBoolVar(f"b_{dep.src}_{dep.dst}_{da}_{db}")
                    model.Add(b <= x[(dep.src, da)])
                    model.Add(b <= x[(dep.dst, db)])
                    model.Add(b >= x[(dep.src, da)] + x[(dep.dst, db)] - 1)
                    tc = self.transfer_cost(da, db, dep.bytes_mb)
                    model.Add(s[dep.dst] >= e[dep.src] + tc).OnlyEnforceIf(b)

        objective = []
        for t in T:
            task = self.model.tasks[t]
            objective.append(task.priority * e[t])
            if relax_deadline:
                # Penalty for delay if we exceed the deadline
                # and
                late_penalty = model.NewIntVar(0, horizon, f"late_{t}")
                model.Add(late_penalty >= e[t] - task.deadline)
                model.Add(late_penalty >= 0)
                objective.append(10000 * late_penalty)
            for d in D:
                dev = self.model.devices[d]
                base = max(1, int(round(1000 * task.compute / max(dev.speed, 1e-9))))
                objective.append(
                    (base + (0 if not task.preferred_kind or task.preferred_kind == dev.kind else 3000))
                    * x[(t, d)]
                )

        model.Minimize(sum(objective))

        for key, value in hints.items():
            if key[0] == "x":
                t, d = key[1], key[2]
                if (t, d) in x:
                    model.AddHint(x[(t, d)], int(value))
            elif key[0] == "s" and key[1] in s:
                model.AddHint(s[key[1]], int(value))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 15
        solver.parameters.num_search_workers = 8

        st = solver.Solve(model)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        rows = []
        for t in T:
            dsel = next(d for d in D if solver.Value(x[(t, d)]) == 1)
            rows.append({
                "task": t,
                "device": dsel,
                "kind": self.model.devices[dsel].kind,
                "start": solver.Value(s[t]),
                "end": solver.Value(e[t]),
                "dur": solver.Value(dur[t]),
                "deadline": self.model.tasks[t].deadline,
                "priority": self.model.tasks[t].priority,
            })
        import pandas as pd
        return pd.DataFrame(rows)
        
        
    def _heuristic_plan(self, horizon):
        # Very simple plan: for each task, choose the "best" device,
        # then run them sequentially on this device, ignoring dependencies.
        # ...

        rows = []
        now = 0
        for t in self.model.tasks:
            task = self.model.tasks[t]
            dev_name = self.best_device_for(task)
            dev = self.model.devices[dev_name]
            duration = max(1, int(round(1000 * task.compute / max(dev.speed, 1e-9))))
            start = now
            end = min(horizon, start + duration)
            rows.append({
                "task": t,
                "device": dev_name,
                "kind": dev.kind,
                "start": start,
                "end": end,
                "dur": end - start,
                "deadline": task.deadline,
                "priority": task.priority,
            })
            now = end
            if now >= horizon:
                break
        import pandas as pd
        if not rows:
            return pd.DataFrame(columns=[
                "task", "device", "kind", "start", "end", "dur", "deadline", "priority"
            ])
        return pd.DataFrame(rows)
        
        
    def make_segment(self, task_name, now, device, migrated=False, resumed=False):
        task = self.model.tasks[task_name]
        remaining = self.remaining_compute(task_name)
        if remaining <= 0:
            return None

        dev = self.model.devices[device]
        seg_compute = min(remaining, max(1, int(dev.speed * task.segment_ms / 1000)))
        duration = max(1, int(round(1000 * seg_compute / max(dev.speed, 1e-9))))

        return PlannedSegment(
            task=task.name,
            segment_id=task.progress // max(1, seg_compute),
            device=device,
            start=now,
            end=now + duration,
            duration=duration,
            compute=seg_compute,
            migrated=migrated,
            resumed=resumed or (task.progress > 0),
        )
        
    def apply_segment(self, seg):
        # Apply a scheduled segment to the model (advance the task progress).
        task = self.model.tasks[seg.task]
        task.progress = min(task.compute, task.progress + seg.compute)
        task.current_device = seg.device

    def reset_tasks(self):
        """
        Reset per-episode task state (progress and current_device).
        """
        for task in self.model.tasks.values():
            task.progress = 0
            task.current_device = ""