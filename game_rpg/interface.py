"""
TUI Interface for Research RPG.
"""

import time
import sys
from .core import ResearchGame

class GameInterface:
    def __init__(self):
        self.game = ResearchGame()

    def run(self):
        """
        Main Game Loop.
        """
        print("\n" + "="*50)
        print(" 🧪  WELCOME TO THE BIOPLAUSIBLE LAB RPG 🧪 ")
        print("="*50 + "\n")

        while True:
            # Main Menu
            print(f"\n📊 Status: Level {self.game.level} | XP: {self.game.xp} | SP: {self.game.science_points:.1f}")

            # Show active quest summary
            active_quests = [q for q in self.game.stats.get("quests", []) if not q["completed"]]
            if active_quests:
                print(f"📜 Active Quests: {len(active_quests)}")

            print("\nSelect an action:")
            print("  1. Plan Next Experiment 📋")
            print("  2. Enable Auto-Scientist Mode 🤖")
            print("  3. Visit Upgrade Shop 🛒")
            print("  4. View Quests 📜")
            print("  5. View Lab Status 📈")
            print("  6. Exit 🚪")

            choice = input("\n> ").strip()

            if choice == "1":
                self.menu_plan_experiment()
            elif choice == "2":
                self.menu_auto_mode()
            elif choice == "3":
                self.menu_shop()
            elif choice == "4":
                self.menu_quests()
            elif choice == "5":
                self.menu_status()
            elif choice == "6":
                print("\n👋 Goodbye, Scientist!")
                break
            else:
                print("Invalid choice.")

    def menu_plan_experiment(self):
        candidates = self.game.get_available_experiments()
        if not candidates:
            print("\n⚠️  No available experiments! Try waiting or check resources.")
            return

        print("\n📋 Available Experiments:")
        display_count = min(5, len(candidates))
        for i in range(display_count):
            task = candidates[i]
            print(f"  {i+1}. [{task.tier.name}] {task.model_name} on {task.task_name} (Priority: {task.priority:.1f})")

        print(f"  {display_count+1}. Cancel")

        sub_choice = input("\nSelect Experiment > ").strip()
        try:
            idx = int(sub_choice) - 1
            if 0 <= idx < display_count:
                self.game.execute_task(candidates[idx])
            else:
                print("Cancelled.")
        except ValueError:
            print("Invalid selection.")

    def menu_auto_mode(self):
        print("\n🤖 Auto-Scientist Enabled. Press Ctrl+C to stop.")

        # Check for auto speed upgrade
        speed_mult = self.game.upgrade_manager.get_multiplier("auto_speed")
        delay = 2.0 / speed_mult

        try:
            while True:
                # 1. Plan
                candidates = self.game.get_available_experiments()
                if not candidates:
                    print("   💤 No viable experiments found. Lab is idle.")
                    time.sleep(delay)
                    continue

                task = candidates[0]
                print("\n🤖 Auto-Scientist is thinking...")
                time.sleep(delay / 2)

                self.game.execute_task(task)
                time.sleep(delay)

        except KeyboardInterrupt:
            print("\n🛑 Auto-Mode Stopped.")

    def menu_shop(self):
        while True:
            sp = self.game.science_points
            print(f"\n🛒 UPGRADE SHOP (SP: {sp:.1f})")
            upgrades = self.game.upgrade_manager.get_all_upgrades()

            for i, u in enumerate(upgrades):
                lvl = self.game.upgrade_manager.get_level(u.id)
                cost = self.game.upgrade_manager.get_cost(u.id)

                cost_str = f"{cost:.1f} SP" if cost < float('inf') else "MAXED"
                print(f"  {i+1}. {u.name} (Lvl {lvl}) - {cost_str}")
                print(f"     {u.description}")

            print(f"  {len(upgrades)+1}. Back")

            choice = input("\nBuy Upgrade > ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(upgrades):
                    u = upgrades[idx]
                    if self.game.upgrade_manager.purchase(u.id):
                        print(f"✅ Purchased {u.name}!")
                    else:
                        print("❌ Not enough Science Points.")
                elif idx == len(upgrades):
                    break
            except ValueError:
                pass

    def menu_quests(self):
        quests = self.game.stats.get("quests", [])
        print("\n📜 QUEST LOG")

        if not quests:
            print("  No active quests.")
            # Option to refresh?
            print("  (Quests refresh automatically)")

        for q in quests:
            status = "✅ DONE" if q["completed"] else "IN PROGRESS"
            if not q["completed"]:
                if q["target_type"] == "run_count":
                    status = f"{q['progress']}/{q['target_count']}"
                elif q["target_type"] == "accuracy":
                    status = f"Target: >{q['target_value']:.1%}"

            print(f"  - {q['description']}")
            print(f"    Status: {status} | Reward: {q['reward_xp']} XP, {q['reward_sp']} SP")

        input("\nPress Enter...")

    def menu_status(self):
        print(f"\n📈 Lab Status Report")
        print(f"  - Level: {self.game.level}")
        print(f"  - Total XP: {self.game.xp}")
        print(f"  - Next Level at: {self.game.LEVEL_THRESHOLDS.get(self.game.level + 1, 'MAX')}")
        print(f"  - Science Points: {self.game.science_points:.2f}")
        print(f"  - Experiments Run: {self.game.stats.get('experiments_run', 0)}")

        print("\n  Active Effects:")
        print(f"  - XP Gain: x{self.game.upgrade_manager.get_multiplier('xp_gain'):.2f}")
        print(f"  - SP Gain: x{self.game.upgrade_manager.get_multiplier('sp_gain'):.2f}")
        print(f"  - Auto Speed: x{self.game.upgrade_manager.get_multiplier('auto_speed'):.2f}")

        input("\nPress Enter to continue...")
