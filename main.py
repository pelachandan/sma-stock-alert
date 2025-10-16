from utils.scanner import run_scan
from utils.email_utils import send_email_alert

if __name__ == "__main__":
    print("🚀 Running SMA crossover and 52-week high scan...")
    sma_list, high_list = run_scan()

    print("\n📈 SMA Signals:")
    for s in sma_list:
        print(
            f"  {s['ticker']} - {s['PctAbove']}% above crossover "
            f"(Crossed on {s['CrossoverDate']})"
        )

    print("\n🏆 New Highs:", high_list or "None")

    send_email_alert(sma_list, high_list)
