from utils.scanner import run_scan
from utils.email_utils import send_email_alert

if __name__ == "__main__":
    print("🚀 Running SMA crossover and 52-week high scan...")
    sma_list, high_list = run_scan()

    print("\n📈 SMA Signals:", sma_list)
    print("🏆 New Highs:", high_list)

    # Combined summary email
    summary = ""
    if sma_list:
        summary += "SMA Crossovers:\n" + "\n".join(sma_list) + "\n\n"
    if high_list:
        summary += "New 52-week Highs:\n" + "\n".join(high_list)

    send_email_alert([], subject_prefix="Market Summary", custom_body=summary or "No signals today.")
