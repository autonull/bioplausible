"""
Research Console Interface.
"""

import time
import sys
from .core import ResearchGame
from bioplausible.hyperopt import PatientLevel

class GameInterface:
    def __init__(self):
        self.game = ResearchGame()

    def run(self):
        """
        Main Loop.
        """
        print("\n" + "="*50)
        print(" 🔬  BIOPLAUSIBLE RESEARCH CONSOLE  🔬 ")
        print("="*50 + "\n")

        while True:
            # Dashboard Header
            top_findings = self.game.get_top_discoveries(1)
            best_acc = top_findings[0].accuracy if top_findings else 0.0
            best_model = top_findings[0].model_name if top_findings else "N/A"

            print(f"\n📊 Session Exp: {self.game.stats['experiments_run_session']} | Best Finding: {best_model} ({best_acc:.2%})")

            print("\nCOMMANDS:")
            print("  1. [P]lan Experiments")
            print("  2. [A]uto-Pilot Mode")
            print("  3. [R]eview Discoveries")
            print("  4. [E]xit")

            choice = input("\n> ").strip().lower()

            if choice in ["1", "p"]:
                self.menu_plan()
            elif choice in ["2", "a"]:
                self.menu_auto()
            elif choice in ["3", "r"]:
                self.menu_discoveries()
            elif choice in ["4", "e", "exit", "quit"]:
                print("\n👋 Closing Research Console.")
                break
            else:
                print("Invalid command.")

    def menu_plan(self):
        candidates = self.game.get_available_experiments()
        if not candidates:
            print("\n⚠️  No hypotheses available. Lab is analyzing data...")
            return

        print("\n📋 PROPOSED HYPOTHESES:")
        print(f"   {'PRIORITY':<10} | {'TIER':<10} | {'MODEL':<20} | {'TASK':<10}")
        print("-" * 60)

        display_count = min(10, len(candidates))
        for i in range(display_count):
            task = candidates[i]
            prio_str = f"{task.priority:.1f}"
            print(f" {i+1}. {prio_str:<9} | {task.tier.name:<10} | {task.model_name:<20} | {task.task_name:<10}")

        print(f" {display_count+1}. Cancel")

        sub_choice = input("\nSelect Hypothesis to Test > ").strip()
        try:
            idx = int(sub_choice) - 1
            if 0 <= idx < display_count:
                self.game.execute_task(candidates[idx])
            else:
                print("Cancelled.")
        except ValueError:
            print("Invalid selection.")

    def menu_auto(self):
        print("\n🤖 AUTO-PILOT ENGAGED. Monitoring system...")
        print("Press Ctrl+C to override manual control.")

        try:
            while True:
                # 1. Plan
                candidates = self.game.get_available_experiments()
                if not candidates:
                    print("   💤 No hypotheses. Standby...")
                    time.sleep(5)
                    continue

                task = candidates[0]
                print(f"\n⚡ Executing High-Priority Task: {task.model_name} [{task.tier.name}]")
                time.sleep(1)

                self.game.execute_task(task)
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Auto-Pilot Disengaged.")

    def menu_discoveries(self):
        findings = self.game.get_top_discoveries(10)
        if not findings:
            print("\n📭 No discoveries recorded yet.")
            return

        print("\n🏆 TOP DISCOVERIES:")
        print(f"   {'ACCURACY':<10} | {'MODEL':<20} | {'TASK':<10} | {'PARAMS'}")
        print("-" * 70)

        for i, f in enumerate(findings):
            # Params count might be in trial.params or trial.user_attrs?
            # Metric object from HyperoptStorage is a custom dataclass usually
            # But get_all_trials returns a list of objects with .accuracy, .model_name, etc.
            acc = f.accuracy
            print(f" {i+1}. {acc:<9.2%} | {f.model_name:<20} | {f.config.get('task','?'):<10}")

        input("\nPress Enter to return...")
