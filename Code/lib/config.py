"""
Shared constants for Stage 4 assembly. Imported by all five notebooks so they
cannot drift apart.
"""
from pathlib import Path

ROOT     = Path('../../../Data/Data_Collection/Final')
AGG_Z    = ROOT / 'Stage_4_Normalised'
PANEL_Z  = ROOT / 'Stage_4_Normalised_Panel'
OUT      = ROOT / 'Stage_5_Model_Ready'

# 2007-07-30 is the latest per-feature warm-up end (CFTC weekly). Starting
# earlier means the 22 CFTC features get ~590 rows of fabricated "positioning
# was exactly average". Split A's training window is still 2007-08 to 2015-12.
START_DATE = '2007-08-01'
END_DATE   = '2024-12-31'

CLIP  = 5.0
DTYPE = 'float32'          # PyTorch's default; float64 doubles memory for nothing

# Gap repair at NATIVE cadence, before expanding to daily. Up to three periods.
FFILL_LIMIT = {'daily': 5, 'weekly': 9, 'monthly': 3}

# Expansion to the daily calendar. Backstop only -- the ffill above has already
# repaired anything shorter. Prevents merge_asof reaching back arbitrarily far
# and pairing a daily row with a value from months earlier.
ASOF_TOL = {'weekly': '25D', 'monthly': '100D', 'panel_monthly': '100D'}

CRASH_THRESH = -2.0        # percent; y_binary = minret_5d_pct < CRASH_THRESH

# Columns that must survive the union. None of these has a row in
# decisions_*.csv, because normalise.py never saw them -- so nothing in the
# drop list protects them and a naive column selection loses them silently.
META = {
    'agg_market_daily_means':         ['date', 'target_daily_return'],
    'agg_market_daily_full_moments':  ['date', 'target_daily_return'],
    'weekly_raw':                     ['date'],
    'agg_market_monthly_means':       ['date', 'target_monthly_return'],
    'agg_market_monthly_full_moments':['date', 'target_monthly_return'],
    'panel_stock_daily_engineered':   ['permno', 'date', 'dlyret', 'dlycap'],
    'panel_stock_monthly_engineered': ['permno', 'date', 'month_end_cap'],
}

# Binary regime indicators: passed through un-z-scored, exempt from all rules,
# and absent from the diagnostic report for the same reason.
BINARIES = ['vix_above_20', 'vix_above_30', 'curve_inverted_2y10y',
            'curve_inverted_3m10y', 'credit_stress']