//------------------------------------------------------------------------------
//
// EAS 510
// Fall 2026
// Project 2 data cleaning & prep
// Github repository: eas510-project2-ianroma
// 
//------------------------------------------------------------------------------


clear all

cd "/Users/icrom/Desktop/UB Master/Projects/510_dec_tree"

// create new folders + remove any error
capture mkdir "data/clean"
capture mkdir "data/sample"
capture mkdir "results"

// load variables 
use year case_id st citizen vv_turnout_gvm vv_turnout_pvm   ///
    voted_pres_08 voted_pres_12 voted_pres_16 voted_pres_20  ///
    age educ faminc race ownhome newsint pid3  ///
    using "data/raw/cumulative_2006-2025.dta", clear
	
// clean the data
	// keep the eight even-year elections 2010-2024; 2010 is the first year with 
	// both voting-history parts; 2024 is the latest year with validated turnout 
	keep if inrange(year, 2010, 2024) & mod(year, 2) == 0
	
	// drop anyone without a validated turnout record (target group)
	drop if missing(vv_turnout_gvm)
	
	// drop LA ... its primary is held on election day, so theres no way to 
	// detect effects before nov 
	decode st, generate(state)
	drop if state == "LA"

	// drop non-citizens
	drop if citizen ==2

// generate target - did official voter records confirm a vote in the general?
	// i.e., verified vote
	generate voted = (vv_turnout_gvm ==1)
	
// generate voting history
	// primary
	generate voted_primary = (vv_turnout_pvm ==2)
	
	// previous presidential cycles
	generate voted_prior_pres = 0
	replace voted_prior_pres = 1 if inlist(year, 2010, 2012) & !missing(voted_pres_08) & voted_pres_08 != 4
	replace voted_prior_pres = 1 if inlist(year, 2014, 2016) & !missing(voted_pres_12) & voted_pres_12 != 5
	replace voted_prior_pres = 1 if inlist(year, 2018, 2020) & !missing(voted_pres_16) & voted_pres_16 != 8
	replace voted_prior_pres = 1 if inlist(year, 2022, 2024) & !missing(voted_pres_20) & voted_pres_20 != 7
	
	// 0-2 history score
	generate vote_history = voted_primary + voted_prior_pres
		// 2 = voted in both primary and general
		// 1 = voted in only one of the two
		// 0 = voted in neither 
		
// curating features for python model
	// ordered categories stay numbers, so the tree can ask "at most X?"
		// educ: already 1 = no HS, through 6 = post grad
		// faminc: 1-12, but 13 "prefer not to say" and 14 "skipped" become missing
		// Python fills them in, using training data only, so remove 
		generate income = faminc if faminc <= 12
		// newsint: flip so a bigger number = more interest + "don't" know is missing
		generate news_interest = 5 - newsint if newsint <= 4
		
	// unordered categories become text (Python will one-hot encode later)
	decode race, generate(race_group)
	decode ownhome, generate(housing)
	decode pid3, generate(party)
	
// keep final variables 
keep year case_id voted voted_primary voted_prior_pres vote_history ///
     age educ income news_interest race_group housing party state
order year case_id voted voted_primary voted_prior_pres vote_history ///
     age educ income news_interest race_group housing party state	 
sort year case_id

// exort full dataset
export delimited using "data/clean/ces_clean_full.csv", replace nolabel

// export small sample for github repo: 2000 random respondents from each year (16k total)
set seed 42
sample 2000, count by(year)
sort year case_id
export delimited using "data/sample/ces_clean_sample.csv", replace nolabel
	
beep
