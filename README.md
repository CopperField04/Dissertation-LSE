# Data Access Link

Link to Google Drive: https://drive.google.com/drive/folders/1msxkUoMaf33rgNaWII8OTe28iITsljAy?usp=sharing

# Full Tree of Local Code and Data

C:.
├───Code
│   ├───Data_Cleaning
│   │   ├───01_Top100_SP500_Universe_Cleaned
│   │   ├───02_FRED_Data_Cleaning
│   │   ├───03_AAII_Sentiment_Cleaning
│   │   ├───04_LSEG_IBES_Cleaning
│   │   ├───05_CTFC_Cleaning
│   │   ├───06_Daily_CRSP_Stock_Data_Cleaning
│   │   ├───07_OpenAssetPricing_Cleaning
│   │   ├───08_TAQ_Millisecond_Cleaning
│   │   ├───09_Macro_Daily_Monthly_WRDS_Cleaning
│   │   ├───10_Options_Data_Cleaning
│   │   └───11_VIX_Futures_CBOE_Skew_Cleaning.ipynb
│   ├───Data_Collection
│   ├───Data_Merging
│   │   ├───OLD_Stage_3_Model_Ready
│   │   │   ├───feature_naming
│   │   │   ├───full_moments
│   │   │   ├───means_only
│   │   │   └───theme_allocation
│   │   ├───Stage_1
│   │   ├───Stage_1_5_Validation_and_Feature_Engineering_Code
│   │   ├───Stage_2
│   │   ├───Stage_3_Cleaning
│   │   ├───Stage_4_Assembly
│   │   └───Stage_5_Themes
│   ├───KAN_Section
│   │   ├───Dense_vs_Sparse_KAN
│   │   │   └───__pycache__
│   │   └───Panel_KAN
│   ├───lib
│   │   └───__pycache__
│   ├───OLD_Data_Panel_Creation
│   ├───OLD_Data_Train_Val_Test_Splitting
│   └───Polymodel
├───Data
│   ├───Data_Collection
│   │   ├───Cleaned
│   │   │   ├───01_Top100_SP500_Universe
│   │   │   ├───02_FRED
│   │   │   ├───03_AAII_Sentiment
│   │   │   ├───04_LSEG_IBES
│   │   │   ├───05_CFTC
│   │   │   ├───06_Daily_CRSP_Stock_Data
│   │   │   ├───07_OpenAssetPricing
│   │   │   ├───08_TAQ_Millisecond
│   │   │   ├───09_Macro_Daily_Monthly_WRDS
│   │   │   ├───10_OptionMetrics
│   │   │   └───11_VIX_SKEW
│   │   ├───Final
│   │   │   ├───Stage_1_5_Validation_and_Feature_Engineering
│   │   │   ├───Stage_1_Initial_Merge
│   │   │   ├───Stage_2
│   │   │   ├───Stage_3_Cleaning
│   │   │   ├───Stage_3_Model_Ready
│   │   │   │   └───themes
│   │   │   ├───Stage_3_Normalisation
│   │   │   │   └───01_excluded
│   │   │   ├───Stage_4_Final_w_Calendar_Theme_Removed
│   │   │   │   └───themes
│   │   │   ├───Stage_4_Normalised
│   │   │   ├───Stage_4_Normalised_Panel
│   │   │   └───Stage_5_Model_Ready
│   │   │       ├───01_unioned
│   │   │       ├───02_assembled
│   │   │       ├───03_targets
│   │   │       ├───04_splits
│   │   │       │   ├───Split_A
│   │   │       │   ├───Split_B
│   │   │       │   ├───Split_C
│   │   │       │   └───Split_D
│   │   │       └───05_themes
│   │   │           ├───by_subtheme
│   │   │           ├───by_theme
│   │   │           └───shape_subthemes
│   │   └───Initial
│   │       ├───01_CTFC
│   │       ├───02_FRED
│   │       ├───03_AAII_Sentiment
│   │       ├───04_LSEG_IBES
│   │       ├───05_Top100_SP500_Universe
│   │       ├───06_Daily_CRSP_Stock_Data
│   │       │   └───firm_daily
│   │       │       ├───year=2004
│   │       │       ├───year=2005
│   │       │       ├───year=2006
│   │       │       ├───year=2007
│   │       │       ├───year=2008
│   │       │       ├───year=2009
│   │       │       ├───year=2010
│   │       │       ├───year=2011
│   │       │       ├───year=2012
│   │       │       ├───year=2013
│   │       │       ├───year=2014
│   │       │       ├───year=2015
│   │       │       ├───year=2016
│   │       │       ├───year=2017
│   │       │       ├───year=2018
│   │       │       ├───year=2019
│   │       │       ├───year=2020
│   │       │       ├───year=2021
│   │       │       ├───year=2022
│   │       │       ├───year=2023
│   │       │       └───year=2024
│   │       ├───07_OpenAssetPricing
│   │       │   └───firm_monthly
│   │       │       ├───year=2004
│   │       │       ├───year=2005
│   │       │       ├───year=2006
│   │       │       ├───year=2007
│   │       │       ├───year=2008
│   │       │       ├───year=2009
│   │       │       ├───year=2010
│   │       │       ├───year=2011
│   │       │       ├───year=2012
│   │       │       ├───year=2013
│   │       │       ├───year=2014
│   │       │       ├───year=2015
│   │       │       ├───year=2016
│   │       │       ├───year=2017
│   │       │       ├───year=2018
│   │       │       ├───year=2019
│   │       │       ├───year=2020
│   │       │       ├───year=2021
│   │       │       ├───year=2022
│   │       │       ├───year=2023
│   │       │       └───year=2024
│   │       ├───08_TAQ_Millisecond
│   │       │   └───firm_daily_taq
│   │       │       ├───year=2004
│   │       │       ├───year=2005
│   │       │       ├───year=2006
│   │       │       ├───year=2007
│   │       │       ├───year=2008
│   │       │       ├───year=2009
│   │       │       ├───year=2010
│   │       │       ├───year=2011
│   │       │       ├───year=2012
│   │       │       ├───year=2013
│   │       │       ├───year=2014
│   │       │       ├───year=2015
│   │       │       ├───year=2016
│   │       │       ├───year=2017
│   │       │       ├───year=2018
│   │       │       ├───year=2019
│   │       │       ├───year=2020
│   │       │       ├───year=2021
│   │       │       ├───year=2022
│   │       │       ├───year=2023
│   │       │       └───year=2024
│   │       ├───09_Macro_Daily_Monthly_WRDS
│   │       ├───10_OptionSuite
│   │       │   ├───final
│   │       │   ├───om_borrow_rates_yearly
│   │       │   ├───om_greeks_positioning_yearly
│   │       │   └───om_vol_surface_yearly
│   │       └───11_VIX_Futures_and_CBOE_SKEW
│   ├───Diagnostics
│   │   └───panel_preclip
│   ├───Polymodel
│   ├───Results
│   │   ├───Dense_vs_Sparse_KAN
│   │   │   ├───final_tables_baselines
│   │   │   │   ├───csv
│   │   │   │   └───latex
│   │   │   ├───Polymodel
│   │   │   │   ├───diagnostics
│   │   │   │   ├───metrics
│   │   │   │   └───predictions
│   │   │   ├───Ridge
│   │   │   │   ├───metrics
│   │   │   │   └───predictions
│   │   │   └───Volatility_Baseline
│   │   └───Reproducing_Hellinger_Polymodel
│   │       ├───figures
│   │       ├───intermediate
│   │       └───tables
│   ├───Splits
│   │   ├───Split_A
│   │   ├───Split_B
│   │   ├───Split_C
│   │   ├───Split_D
│   │   └───themes
│   └───Splits_Panel
│       ├───Split_A
│       ├───Split_B
│       ├───Split_C
│       └───Split_D
├───Figures
│   └───Polymodel
└───Writeup
    ├───.vscode
    ├───chapters
    ├───figures
    │   ├───conclusion
    │   ├───data
    │   ├───empirical
    │   ├───introduction
    │   ├───kan_framework
    │   ├───literature
    │   └───polymodels
    └───tables
