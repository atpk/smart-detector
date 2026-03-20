"""
alert_system.py
───────────────
Handles two categories of alerts:
  1. Fall alerts  — confirmed fall event
  2. Vitals alerts — abnormal HR, SpO₂, RR readings
Both use individual cooldown timers and log to alerts.log.
"""

import os
import sys
import time
import datetime

import config


class AlertSystem:
    """
    Checks fall results and vitals results each frame and fires
    alerts with cooldown gating.
    """

    def __init__(self):
        self._fall_last_alert  = 0.0
        self._vitals_cooldowns = {}    # key → last alert epoch
        self._alert_count      = 0
        self._log_path         = config.ALERT_LOG_FILE
        self._ensure_log()
        print(f"[AlertSystem] Alert log: {os.path.abspath(self._log_path)}")

    # ── Public ────────────────────────────────────────────────────────────

    def check(self, fall_result, vitals_result=None) -> dict:
        """
        Evaluate both results and fire any triggered alerts.
        Returns dict of {alert_type: bool} for this call.
        """
        fired = {}

        # Fall alert
        if fall_result.fall_confirmed:
            if self._cooldown_ok("fall", config.ALERT_COOLDOWN_SEC):
                self._fire_fall(fall_result)
                fired["fall"] = True

        # Vitals alerts
        if vitals_result is not None and vitals_result.ready:
            vr = vitals_result

            if vr.heart_rate is not None:
                if vr.heart_rate < config.ALERT_HR_LOW:
                    if self._cooldown_ok("hr_low", 30):
                        self._fire_vitals("Low Heart Rate",
                                          f"{vr.heart_rate:.0f} bpm (< {config.ALERT_HR_LOW})")
                        fired["hr_low"] = True
                elif vr.heart_rate > config.ALERT_HR_HIGH:
                    if self._cooldown_ok("hr_high", 30):
                        self._fire_vitals("High Heart Rate",
                                          f"{vr.heart_rate:.0f} bpm (> {config.ALERT_HR_HIGH})")
                        fired["hr_high"] = True

            if vr.spo2 is not None and vr.spo2 < config.ALERT_SPO2_LOW:
                if self._cooldown_ok("spo2", 30):
                    self._fire_vitals("Low SpO₂",
                                      f"{vr.spo2:.1f}% (< {config.ALERT_SPO2_LOW}%)")
                    fired["spo2"] = True

            if vr.resp_rate is not None and vr.resp_rate > config.ALERT_RR_HIGH:
                if self._cooldown_ok("rr_high", 30):
                    self._fire_vitals("High Respiratory Rate",
                                      f"{vr.resp_rate:.0f} br/min (> {config.ALERT_RR_HIGH})")
                    fired["rr_high"] = True

        return fired

    @property
    def alert_count(self) -> int:
        return self._alert_count

    # ── Private ───────────────────────────────────────────────────────────

    def _cooldown_ok(self, key: str, seconds: float) -> bool:
        now = time.time()
        if (now - self._vitals_cooldowns.get(key, 0.0)) >= seconds:
            self._vitals_cooldowns[key] = now
            return True
        return False

    def _fire_fall(self, result):
        self._alert_count += 1
        ts        = self._ts()
        score_pct = round(result.score * 100, 1)
        msg = (
            f"\n\033[91m{'─' * 52}\n"
            f"  ⚠  FALL DETECTED  — Alert #{self._alert_count}\n"
            f"  Time  : {ts}\n"
            f"  Score : {score_pct}%   Streak: {result.fall_streak} frames\n"
            f"{'─' * 52}\033[0m\n"
        )
        print(msg, flush=True)
        self._beep()
        sub = ", ".join(f"{k}={v:.2f}" for k, v in result.sub_scores.items())
        self._log(f"[{ts}]  FALL        score={score_pct}%  streak={result.fall_streak}  ({sub})")

    def _fire_vitals(self, alert_name: str, detail: str):
        self._alert_count += 1
        ts = self._ts()
        msg = (
            f"\n\033[93m{'─' * 52}\n"
            f"  ⚡  VITALS ALERT: {alert_name}\n"
            f"  Time  : {ts}\n"
            f"  Value : {detail}\n"
            f"{'─' * 52}\033[0m\n"
        )
        print(msg, flush=True)
        self._beep()
        self._log(f"[{ts}]  VITALS      {alert_name}: {detail}")

    def _beep(self):
        if config.ALERT_BEEP:
            sys.stdout.write("\a")
            sys.stdout.flush()

    def _log(self, line: str):
        try:
            with open(self._log_path, "a") as f:
                f.write(line + "\n")
        except OSError as e:
            print(f"[AlertSystem] Log write error: {e}")

    def _ensure_log(self):
        if not os.path.exists(self._log_path):
            try:
                with open(self._log_path, "w") as f:
                    f.write("# Health Monitor Alert Log\n")
            except OSError:
                pass

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
