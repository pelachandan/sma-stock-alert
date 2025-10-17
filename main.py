from utils.scanner import run_scan
from utils.email_utils import send_email_alert

if __name__ == "__main__":
    print("🚀 Running EMA crossover and 52-week high scan...")
    ema_list, high_list = run_scan(test_mode=True)
    
    for s in ema_list:
        print(
            f"  {s['ticker']} - {s['PctAbove']}% above crossover "
            f"(Crossed on {s['CrossoverDate']})"
        )

    print("\n🏆 New Highs:", high_list or "None")

    send_email_alert(ema_list, high_list)
