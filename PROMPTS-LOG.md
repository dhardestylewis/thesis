Thesis ongoing prompts and notes, in part to guide any future LLM on every aspect and consideration which helps prevents detail-regressions
==========================================











































make 

REVISION 2: Task Schedule (condensed to 1 page)

6. Task Schedule

Phase I: Literature Review and Data Acquisition (October–November 2025)

Complete comprehensive literature review: October 1–November 15

Obtain IRB approval: Submit November 25, anticipate approval by mid-December

Acquire datasets via public information requests and open data portals: October 15–December 15

Milestone: Complete thesis proposal (December 10)

Phase II: Quantitative Analysis (December 2025–January 2026)

Data preparation, cleaning, and linkage: December 15–January 15

Feature construction and exploratory analysis: January 10–January 25

ML model development, validation, and comparison: January 15–February 10

Geographic and temporal analysis: January 20–February 10

Deliverable: Draft quantitative methods and results chapter (February 10)

Phase III: Qualitative Data Collection (January–February 2026)

Finalize interview protocol and recruit participants: January 5–January 25

Conduct interviews (n=10): January 20–February 28

Observe public meetings and collect documentary evidence: January 15–March 15

Transcribe and begin preliminary coding: February 1–March 10

Deliverable: Complete interview transcripts and field notes (March 10)

Phase IV: Integration and Analysis (February–March 2026)

Systematic qualitative coding and thematic analysis: February 20–March 20

Analyze comparative frameworks from other cities: March 1–March 15

Integrate quantitative and qualitative findings: March 10–March 25

Develop theoretical framework connecting to algorithmic governance literature: March 15–March 30

Deliverable: Complete draft chapters submitted to advisor (March 30)

Phase V: Writing and Defense (March–May 2026)

Revise based on advisor feedback: April 1–April 10

Distribute to reader(s): No later than April 3 (14 days before jury)

Prepare jury presentation: April 5–April 15

Thesis jury: April 13–17

Final revisions incorporating jury feedback: April 18–May 5

Final submission: May 8, 2026

even more structured and more consistent with your style. adopt the following style:

I) October–November 2025:

II) December 2025–January 2026:

etc

but use 2025 October-November etc

and within each

October 1–November 15: 

but for these inner ones use 10/01-11/15: etc

do not use —, replace all such instances with single - or , making a written decision for each beforehand outside the artifact

this is not PIR, this is from Austin's Open Data portal

Travis Central Appraisal District records: Parcel-level property characteristics (assessed value, land use, approximate ownership tenure), obtained via public information requests

This is PIRed, not provided in any zoning Open Data

Opposition records: Protest petitions, valid petitions under Texas Local Government Code § 211.006(d), and formal opposition filings extracted from zoning case files

put this at end and label optional

Campaign finance data: Texas Ethics Commission records linking property owners to campaign contributions in city elections

prioritize following order

Opposition records

City of Austin development records

Travis Central Appraisal District records

Demographic context

Public testimony

Campaign finance data

Direct identifiers (names, addresses) will be stored separately from the _published_ - unless this contravenes IRB, and tell me either way - analytic dataset, which will use coded study IDs to minimize privacy risks.



































is retrospective vs prospective a key distinction here? what do you mean by that? drop making the distinction here unless it is worthy enough to elevate to providing some detail in this very first paragraph

"retrospective quantitative analysis of administrative records with prospective qualitative data collection"



more than any of the following is that having conducted a PIR of the top-50 major metros operating under similar state laws, Austin alone has kept such records of protest letters

"Second, Austin maintains comprehensive public records of development cases, zoning protests, and public testimony that enable parcel-level analysis"

the above is first and foremost among reasons



even more valuable is that the period under study includes both periods of zoning stasis and reform, which I agree forms a natural experiment. another notable natural experiment which will not be reflected in the data nor necessarily need be an object of our research and only valuable inasmuch as it structures the 'so what', how what we produce here may be applied, and its own natural experiment immediately pre- and post- is the fact that the law which produced this dataset has since been repealed in Texas (use search to look it up). Again, this is a note for you to keep track of in this conversation, remind me with every prompt turn, and include in the relevant non-research design sections as we iteratively produce not just these sections revise and whittle down the full proposal



identify some widely agreed upon exact point at which dynamic changed from stasis to reform, if any

First, the city has undergone dramatic housing policy reforms between 2023-2025, including the HOME Initiative, elimination of parking requirements, and reduction of minimum lot sizes, creating a natural experiment in how zoning liberalization affects opposition patterns.

otherwise push back and let me know



not for the research design but if we haven't yet included any citations for publications discussing the receptivity in Austin towards these sort of city- or government-spearheaded tech initiatives, the level of tech literacy and competence among citizens, then we should do so



partial regression: 2018-2025 is where the richest data spans, we will not possess differences in parcel and tax variables prior to 2018 as this data wasn't consistently collected annually prior to then, we will hold fixed to the 2018 data all pre-2018 variables, and of course only be evaluating prediction post-2018 to avoid temporal leakage in any backtesting



the same applies to ATXN video recordings, which go back to 2015, we will use all available data, this data will not be filled from post, and will only predict forwards where all data sources begin



Austin Campaign Finance have been digitized back to 2016

Skip to main contentSkip to footer

Data.AustinTexas.gov - The official City of Austin open data portal

Campaign Finance - Transaction Detail

City Government

This table lists individual transactions as reported on the Texas Ethics Commission C/OH Form. Each row represents an individual filer's transaction, which may be a loan, expenditure, credit, pledge or contribution.

For information about each report filed, please visit the Reports Table at: https://data.austintexas.gov/Government/Campaign-Finance-Data-Reports-Table/b2pc-2s8n

A data dictionary for this dataset is available at http://services.austintexas.gov/edims/document.cfm?id=247730

Last Updated

December 10, 2025

Data Provided By

City of Austin, Texas - data.austintexas.gov

About this Dataset

Updated

December 10, 2025

Data Last Updated

December 10, 2025

Metadata Last Updated

December 10, 2025

Date Created

January 14, 2016

Views

28.8K

Downloads

3,464

Data Provided by

City of Austin, Texas - data.austintexas.gov

Dataset Owner

Austin City Clerk's Office Owners

Publishing Information

Update Frequency	As Needed

Ownership

Department name	Austin City Clerk's Office

Topics

Category	City Government

Tags	

campaign finance,city clerk,contributions,election,expenditures,financial

Licensing and Attribution

License	Public Domain

What's in this Dataset?

Rows

255K

Columns

46

Each row is a

Transaction

Row Identifier

Transaction_ID

Columns (46)

Column Name

Description

API Field Name

Data Type

Transaction_Type

Identifies the type of transaction reported: Loan, Expenditure, Credit, Pledge, or Contribution.

transaction_type

Text

Filer_Name

Name of individual, committee or entity filing the report. This name is a standardized format used by the Office of the City Clerk to ensure all of a filer's reports are accessible under a single name

filer_name

Text

Transactor_Name

The transactor's name (for individuals) or the name of the company, political committee, organization, or group (for entities).

transactor_name

Text

Entity

Identifies the transactor as either an entity (such as a company, political committee, organization, or other group) or an individual person

entity

Text

Transaction_Amount

Amount involved in the transaction. Amounts will be listed as a negative (-) for expenditures.

transaction_amount

Number

Date_Filed

Date the structured data file was received by the Office of the City Clerk

date_filed

Floating Timestamp

Form

The type of form filed

form

Text

Report

Unique identifier for the specific report - can be used to link individual transactions to a specific report.

report_id

Text

Transaction_Code

Unique identifier for the individual transaction listed - can be used to link transactions associated with multiple records, e.g. Direct Campaign Expenditures or Out of State Travel

transaction

Text

Transaction_ID

Unique ID based upon the concatenation of the Report_ID and Transaction_ID field - acts as a unique key for each individual transaction

transaction_id

Text

Date_Due

Due date of the report

date_due

Floating Timestamp

Report_Type

The primary report type as listed on the form.

report_type

Text

Election_Date

The date of the election for which this report is filed, if applicable

election_date

Floating Timestamp

Election_Type

The type of the election for which this report is filed, if applicable

election_type

Text

Office_Held

For officeholders, the office currently held

office_held

Text

Rows per page:

15

1-15 of 46

Go to the previous page

Go to the next page

Dataset Changelog

No changes have been archived yet



https://data.austintexas.gov/City-Government/Campaign-Finance-Transaction-Detail/g4yx-aw9r/about_data





































spell out for me the IRB implications of different ways we could handle this as variations from a standard data collection management engineering process pipeline which doesn't have any PI/IRB concerns

These datasets will be linked at the parcel or case level using geocoding and property identifiers. Direct identifiers (names, addresses) will be stored separately from the analytic dataset, which will use coded study IDs to minimize privacy risks.































this detail doesn't belong in this list:

"as a measure of civic engagement with property-related processes."



neither: "(continuous distance measures)", 



































To ensure temporal validity and avoid data leakage, we will train models on 2018-2022 cases and test on 2023-2025 cases.



actually we will do annual expanding window training starting with a training cutoff of 2018 and proceeding each year from there, testing on the next year, and holdout validating any and all remaining years



































check our references from original artifact, did it include the following?



Skip to main contentSkip to footer

Campaign Finance - Transaction Detail

City Government

This table lists individual transactions as reported on the Texas Ethics Commission C/OH Form. Each row represents an individual filer's transaction, which may be a loan, expenditure, credit, pledge or contribution. For information about each report filed, please visit the Reports Table at: https://data.austintexas.gov/Government/Campaign-Finance-Data-Reports-Table/b2pc-2s8n A data dictionary for this dataset is available at http://services.austintexas.gov/edims/document.cfm?id=247730

Last UpdatedDecember 10, 2025

Data Provided ByCity of Austin, Texas - data.austintexas.gov

About this Dataset

Updated

December 10, 2025

Data Last Updated

December 10, 2025

Metadata Last Updated

December 10, 2025

Date Created

January 14, 2016

Views

28.8K

Downloads

3,464

Data Provided by

City of Austin, Texas - data.austintexas.gov

Dataset Owner

Austin City Clerk's Office Owners

Publishing Information

Update FrequencyAs Needed

Ownership

Department nameAustin City Clerk's Office

Topics

CategoryCity GovernmentTags

campaign finance,city clerk,contributions,election,expenditures,financial

Licensing and Attribution

LicensePublic Domain

What's in this Dataset?

Rows

255K

Columns

46

Each row is a

Transaction

Row Identifier

Transaction_ID

Columns (46)

Column Name

Description

API Field NameData Type

Transaction_Type

Identifies the type of transaction reported: Loan, Expenditure, Credit, Pledge, or Contribution.

transaction_type

Text

Filer_Name

Name of individual, committee or entity filing the report. This name is a standardized format used by the Office of the City Clerk to ensure all of a filer's reports are accessible under a single name

filer_name

Text

Transactor_Name

The transactor's name (for individuals) or the name of the company, political committee, organization, or group (for entities).

transactor_name

Text

Entity

Identifies the transactor as either an entity (such as a company, political committee, organization, or other group) or an individual person

entity

Text

Transaction_Amount

Amount involved in the transaction. Amounts will be listed as a negative (-) for expenditures.

transaction_amount

Number

Date_Filed

Date the structured data file was received by the Office of the City Clerk

date_filed

Floating Timestamp

Form

The type of form filed

form

Text

Report

Unique identifier for the specific report - can be used to link individual transactions to a specific report.

report_id

Text

Transaction_Code

Unique identifier for the individual transaction listed - can be used to link transactions associated with multiple records, e.g. Direct Campaign Expenditures or Out of State Travel

transaction

Text

Transaction_ID

Unique ID based upon the concatenation of the Report_ID and Transaction_ID field - acts as a unique key for each individual transaction

transaction_id

Text

Date_Due

Due date of the report

date_due

Floating Timestamp

Report_Type

The primary report type as listed on the form.

report_type

Text

Election_Date

The date of the election for which this report is filed, if applicable

election_date

Floating Timestamp

Election_Type

The type of the election for which this report is filed, if applicable

election_type

Text

Office_Held

For officeholders, the office currently held

office_held

Text

Rows per page:

15

1-15 of 46

Go to the previous page

Go to the next page

Dataset Changelog

No changes have been archived yet

https://data.austintexas.gov/City-Government/Campaign-Finance-Transaction-Detail/g4yx-aw9r/about_data



any and all relevant ones from the following?

Skip to main contentSkip to footer

Data.AustinTexas.gov - The official City of Austin open data portal

Filters

Toggle View Types section

View Types

Calendars

Charts

Datasets

External Datasets

Files and Documents

Filtered Views

Forms

Maps

Measures

Stories

Toggle Category section

Category

Budget and Finance

Building and Development

City Government

City Infrastructure

Environment

Toggle Department section

Department

Austin 311

Austin Animal Services

Austin Arts, Culture, Music, and Entertainment

Austin Aviation

Austin Budget and Organizational Excellence

Toggle Domain section

Domain

data.texas.gov

This site only

Toggle panel

Tags

Choose tag

Catalog

campaign finance

11 results

"campaign finance"

Sort by

Most relevant

Campaign Finance - Transaction Detail

City Government

This table lists individual transactions as reported on the Texas Ethics Commission C/OH Form. Each row represents an individual filer's transaction, which may be a loan, expenditure, credit, pledge or contribution. For information about each report filed, please visit the Reports Table at: https://data.austintexas.gov/Government/Campaign-Finance-Data-Reports-Table/b2pc-2s8n A data dictionary for this dataset is available at http://services.austintexas.gov/edims/document.cfm?id=247730

Tags

expenditures

financial

contributions

city clerk

campaign finance

and 1 more...

Last updated

December 10, 2025

Views

28792

Campaign Finance - Contributions

City Government

This dataset lists all contributions and pledges received by candidates, officeholders and political committees on data files submitted per City Code Chapter 2-2-26. For a complete listing of each column heading, please see the field listing here: http://services.austintexas.gov/edims/document.cfm?id=262464 Note: Texas Election Code § 254.0401 requires the City Clerk to publish campaign finance reports online and was amended in 2023 to allow the Clerk to remove the street address of the contributors listed in the report, as long as the city, state, and ZIP code remain visible in the report, and the entire address is available on the original report maintained by the City Clerk in her office. Due to this change in the law, the City Clerk redacts the street addresses from the online datasets containing the information in campaign finance forms filed with the City Clerk’s office. Original, unredacted reports are maintained at the City Clerk’s office and are available upon request.

Tags

contributions

campaign finance

city clerk

financial

election

Last updated

December 10, 2025

Views

6947

Campaign Finance - Expenditures

City Government

This dataset lists all expenditures and incurred obligations reported by candidates, officeholders and political committees on data files submitted per City Code Chapter 2-2-26. For a complete listing of each column heading, please see the field listing here: http://services.austintexas.gov/edims/document.cfm?id=262465

Tags

city clerk

election

expenditures

campaign finance

financial

Last updated

December 10, 2025

Views

2864

Campaign Finance - Report Detail

City Government

This dataset contains data about the Campaign Finance Report received by a filer. The Reports table contains information about the type of report filed, the due date, original filing date, and transaction totals. To access the Transactions table, where individual transactions are listed, please go to: https://data.austintexas.gov/Government/Campaign-Finance-Transaction-Detail/g4yx-aw9r A data dictionary for this dataset is available at http://services.austintexas.gov/edims/document.cfm?id=247730

Tags

campaign finance

election

expenditures

city clerk

financial

and 1 more...

Last updated

December 10, 2025

Views

2397

Campaign Finance - Credits

City Government

This table lists any credit, interest, rebate, refund, reimbursement, or return of a deposit fee resulting from the use of a political contribution or an asset purchased with a political contribution reported on the Texas Ethics Commission C/OH Form, Schedule K. For more information about credits, please visit the Texas Ethic Commission's instruction guide here: https://www.ethics.state.tx.us/forms/COH_ins.pdf

Tags

financial

election

credits

city clerk

campaign finance

Last updated

December 10, 2025

Views

1291

Campaign Finance - Pre-Election Summary

City Government

A summary of data reported to the Clerk's office within the last 30 days, broken into transaction type and report type, with a transaction total for each type of record (expenditure, contribution, loan) reported. The most recently reported items are at the top of the dataset.

Tags

financial

city clerk

election

campaign finance

Last updated

December 10, 2025

Views

979

Campaign Finance - Loans

City Government

This dataset lists all loans taken out by candidates, officeholders and political committees on data files submitted per City Code Chapter 2-2-26. For a complete listing of each column heading, please see the field listing here: http://services.austintexas.gov/edims/document.cfm?id=262466

Tags

financial

city clerk

loans

election

campaign finance

Last updated

December 10, 2025

Views

1715

Campaign Finance - Direct Campaign Expenditures

City Government

Direct Campaign Expenditures

Tags

expenditures

expenditure

campaign finance

election

city clerk

and 1 more...

Last updated

December 10, 2025

Views

677

Campaign Finance - Travel Outside of Texas

City Government

Contains detailed information for expenditures or an in-kind contributions that were used for travel outside the state of Texas. Data in this sheet corresponds to the information collected on Texas Ethics Commissions's Schedule T.

Tags

campaign finance

contributions

financial

expenditures

election

and 1 more...

Last updated

December 10, 2025

Views

1153

Campaign Finance - Committee Purpose

City Government

If a political committee has supported, opposed, or assisted a candidate, officeholder, or ballot measure during the reporting period, this activity will be reported in the Committee Purpose Dataset.

Tags

political action committee

campaign finance

election

financial

city clerk

Last updated

December 10, 2025

Views

661

PEI 1.2 Average Monthly Youth Served by County, Region, and PEI Program FY2015-2024

Data provided by data.texas.gov

The Division of Prevention and Early Intervention (PEI) was created to consolidate child abuse prevention and juvenile delinquency prevention and early intervention programs within the jurisdiction of a single state agency. Consolidation of these programs is intended to eliminate fragmentation and duplication of contracted prevention and early intervention services for at-risk children, youth, and families: Community Youth Development (CYD) - The CYD program contracts with community-based organizations to develop juvenile delinquency prevention programs in ZIP codes with high juvenile crime rates. Approaches used by communities to prevent delinquency have included mentoring, youth employment programs, career preparation, youth leadership development and recreational activities. Communities prioritize and fund specific prevention services according to local needs. CYD services are available in 15 targeted Texas ZIP codes. Family and Youth Success Program (FAYS) (formerly Services to At-Risk Youth (STAR)) - The FAYS program contracts with community agencies to offer family crisis intervention counseling, short- term emergency respite care, and individual and family counseling. Youth up to age 17 and their families are eligible if they experience conflict at home, truancy or delinquency, or a youth who runs away from home. FAYS services are available in all 254 Texas counties. Each FAYS contractor also provides universal child abuse prevention services, ranging from local media campaigns to informational brochures and parenting classes. Statewide Youth Services Network (SYSN) - The SYSN program contracts provide community and evidence-based juvenile delinquency prevention programs focused on youth ages 10 through 17, in each DFPS region. NOTE: For FY15, as a result of a new procurement, the overall number of youth served decreased however the service requirements were enhanced with additional programmatic components. Data as of December 11, 2024.

Tags

youth

databook

data book

dfps

intervention

and 4 more...

Last updated

January 29, 2025

Views

0

Results per page:

20

1-11 of 11

Go to the first page

Go to the previous page

Go to the next page

Go to the last page

data.austintexas.gov

Home

Data Catalog

Terms of Use

Contact Us

Support

AustinTexas.gov

© 2025 City of Austin, Texas

Tyler Technology logo



a mention of the transcription from these original documents?



Skip to main content

3-1-1

World Icon

Translate

Dropdown Icon

City of Austin - Website

Search

Main menu

Resident

Business

Government

Departments

Connect

Departments City Clerk 2014 - Campaign Finance Reporting

2014 - Campaign Finance Reporting

Home

About

Services

Programs

City Clerk

Committees of Council

Council Corporations

Intergovernmental Entities

Boards and Commissions

Associated Entities

2014 Campaign Finance Reports

Reports Filed by Council Members and Candidates

January 15 / July 15 / 30th Day Before Election / 8th Day Before Election / Runoff / Personal Funds - Loans and Expenditures (Schedule ATX.2) / Pre-Election Report (Schedule ATX.7) / Exceeded $500 Limit / Final

Reports Filed by Committees

January 15 / July 15 / 30th Day Before Election / 8th Day Before Election / Runoff / Special Pre-election Report of Contributions / Special Pre-election Report of Direct Expenditures / Monthly Campaign Finance Report / Dissolution / Independent Expenditures Not By A Candidate (Schedule ATX.1) / Pre-Election Report (Schedule ATX.7) / Exceeded $500 Limit / Other

Reports Filed by Council Members and Candidates

January 15

Filed By

Date Filed

Correction Filed

Cole, Sheryl

1/15/2014

Leffingwell, S. Lee

1/14/2014

Martin, Patsy Woods

1/15/2014

Martinez, Mike

1/15/2014

Morrison, Laura

1/15/2014

Shea, Brigid

1/15/2014

Spelman, William "Bill"

1/14/2014

Tovo, Kathryne Beth

1/13/2014

Wiley, Jay

1/15/2014

Reports Filed by Council Members and Candidates

July 15

Filed By

Date Filed

Correction Filed

Adler, Stephen

7/15/2014

Almanza, Susana R.

7/15/2014

8/15/2014

Ancira, Miguel M.

7/14/2014

Boyt, Jefferson E. "Jeb"

7/15/2014

8/5/2014

Bray, Rebecca A. "Becky"

7/15/2014

9/4/2014

10/3/2014

Bucknall, Andrew J.

7/9/2014

9/11/2014

Buda, Daniel

7/15/2014

Burciaga, Marjorie "Margie"

7/14/2014

7/17/2014

7/29/2014

Cannon, Audrey "Tina"

7/11/2014

Cantu, Mario G.

7/11/2014

8/25/2014

Casar, Gregorio E. "Greg"

7/15/2014

7/31/2014

Cole, Sheryl

7/15/2014

Daniel, Katrina M.

7/15/2014

7/23/2014

Dealey, Amanda "Mandy"

7/15/2014

11/24/2014

Denny, Jason R.

7/15/2014

English, Edwin E. "Ed"

7/8/2014

Fernandez, Julian Limon

7/15/2014

Flannigan, James T. "Jimmy"

7/15/2014

Floyd, William David "Dave"

7/15/2014

Gallo, Sheri P.

7/15/2014

Garza, Delia

7/15/2014

Guajardo, Frank R. "Franko"

7/15/2014

Guzman, Monica A.

7/15/2014

7/16/2014

7/21/2014

Hernandez, Xavier

7/15/2014

Herrin III, Louis C.

7/21/2014

Houston, Ora

7/15/2014

Ingalls, Josiah J.

7/14/2014

Ireland, Shaun D.

7/15/2014

Jacobson, Norman A.

7/15/2014

Kitchen, Ann

7/15/2014

Lamon, Matthew L. "Matt"

7/15/2014

Leffingwell, S. Lee

6/25/2014

Lofton, DeWayne W.

7/15/2014

Lucier, Nicholas Ryan

7/15/2014

Mancillas, Marco

7/15/2014

7/15/2014

Martinez, Mike

7/15/2014

May, Eliza

7/15/2014

Mays, Sharon E.

7/14/2014

McGann, Erin K.

7/12/2014

7/21/2014

McGhee, Fred

7/15/2014

Meeker, Jason W.

7/15/2014

8/18/2014

Menard, Valerie M.

7/15/2014

Morrison, Laura

6/30/2014

Munoz, Manuel A.

7/15/2014

Osemene, Samuel A.

7/15/2014

Paver, James A. "Jimmy"

7/15/2014

Perez Jr., Roberto

7/15/2014

Phelps, Todd H.

7/15/2014

Phillips Jr., Lloyd "Pete"

7/15/2014

Pierce, Darrell

7/15/2014

Pressley, Laura

7/15/2014

Rangel, Eric J.

7/15/2014

Renteria, Sabino "Pio"

7/11/2014

Reyes Jr., Edward A. "Wally"

7/15/2014

8/18/2014

Riley, Chris

7/15/2014

Rodriguez, Luis M. "Mike"

7/15/2014

Rojas, Gabriel "Gabe"

7/18/2014

Salazar, Pete A. (Jr.)

7/14/2014

Scruggs, Edward S.

7/15/2014

Spelman, William "Bill"

7/1/2014

Stephens, Randall F.

7/15/2014

Thomas, Robert D.

7/11/2014

7/15/2014

7/22/2014

Tovo, Kathryne Beth

7/15/2014

10/27/2014

Troxclair, Ellen

7/15/2014

Turullols-Bonilla, Ricardo

7/15/2014

Wiley, Jay

7/15/2014

Worsham, William L. "Bill"

7/15/2014

Zone, Melissa A.

7/15/2014

10/6/2014

Reports Filed by Council Members and Candidates

30th Day Before Election

Filed By

Date Filed

Correction Filed

Adler, Stephen

10/6/2014

Almanza, Susana R.

10/6/2014

Boyt, Jefferson E. "Jeb"

10/6/2014

Bray, Rebecca A. "Becky"

10/6/2014

10/6/2014

Bucknall, Andrew J.

10/3/2014

Buda, Daniel

10/6/2014

10/27/2014

Burciaga, Marjorie "Margie"

10/6/2014

Cannon, Audrey "Tina"

10/6/2014

Cantu, Mario G.

10/6/2014

Cargill, Michael D.

10/6/2014

Casar, Gregorio E. "Greg"

10/6/2014

10/23/2014

Cole, Sheryl

10/6/2014

Daniel, Katrina M.

10/6/2014

10/13/2014

Dealey, Amanda "Mandy"

10/6/2014

Denny, Jason R.

10/6/2014

English, Edwin E. "Ed"

10/6/2014

Fernandez, Julian Limon

10/6/2014

Flannigan, James T. "Jimmy"

10/6/2014

Floyd, William David "Dave"

10/6/2014

Gallo, Sheri P.

10/6/2014

Garza, Delia

10/6/2014

Guzman, Monica A.

10/6/2014

Herrin III, Louis C.

10/6/2014

Hindman, George W.

10/6/2014

Hoerster, Christopher L. "Chris"

10/6/2014

Houston, Ora

10/6/2014

Ingalls, Josiah J.

10/6/2014

Ingraham, Zachary R.

10/6/2014

Ireland, Shaun D.

10/6/2014

Jacobson, Norman A.

10/3/2014

Kelly, Mackenzie

10/6/2014

Kitchen, Ann

10/6/2014

10/21/2014

10/27/2014

Krenek, Mary C.

10/6/2014

Lamon, Matthew L. "Matt"

10/6/2014

Lofton, DeWayne W.

10/6/2014

10/13/2014

Mancillas, Marco

10/6/2014

Martinez, Mike

10/6/2014

May, Eliza

10/6/2014

Mays, Sharon E.

10/6/2014

McGann, Erin K.

10/6/2014

McGhee, Fred

10/6/2014

Meeker, Jason W.

10/6/2014

Menard, Valerie M.

10/6/2014

Orshalick, David M.

10/6/2014

Osemene, Samuel A.

10/6/2014

Paver, James A. "Jimmy"

10/6/2014

Perez Jr., Roberto

10/6/2014

Phelps, Todd H.

10/14/2014

Phillips Jr., Lloyd "Pete"

10/6/2014

Phillips, Kent K.

10/6/2014

Pierce, Darrell

10/6/2014

Pool, Leslie

10/6/2014

10/7/2014

11/7/2014

Pressley, Laura

10/6/2014

Quintero, Jose

10/6/2014

Rangel, Eric J.

10/6/2014

Renteria, Sabino "Pio"

10/6/2014

Reyes Jr., Edward A. "Wally"

10/7/2014

Riley, Chris

10/6/2014

5/6/2015

Rodriguez, Luis M. "Mike"

10/6/2014

Salazar, Pete A. (Jr.)

10/6/2014

Scruggs, Edward S.

10/6/2014

Senecal, David C.

10/6/2014

Sheppard, John C.

10/6/2014

10/7/2014

Stephens, Randall F.

10/6/2014

Stillwell, Matthew D. "Matt"

10/6/2014

Thomas, Robert D.

10/6/2014

Tovo, Kathryne Beth

10/6/2014

10/27/2014

Troxclair, Ellen

10/6/2014

Turullols-Bonilla, Ricardo

10/8/2014

Valera, Jose A.

10/6/2014

Wiley, Jay

10/6/2014

Wittle, Darryl R.

10/6/2014

Worsham, William L. "Bill"

10/6/2014

Zimmerman, Donald S. "Don"

10/6/2014

Zone, Melissa A.

10/6/2014

Reports Filed by Council Members and Candidates

8th Day Before Election

Filed By

Date Filed

Correction Filed

Adler, Stephen

10/27/2014

Almanza, Susana R.

10/27/2014

Boyt, Jefferson E. "Jeb"

10/27/2014

Bray, Rebecca A. "Becky"

10/27/2014

11/4/2014

Bucknall, Andrew J.

10/27/2014

Buda, Daniel

10/27/2014

Burciaga, Marjorie "Margie"

10/27/2014

Cannon, Audrey "Tina"

10/27/2014

Cantu, Mario G.

10/27/2014

Cargill, Michael D.

10/27/2014

Casar, Gregorio E. "Greg"

10/27/2014

Cole, Sheryl

10/27/2014

Daniel, Katrina M.

10/27/2014

Dealey, Amanda "Mandy"

10/27/2014

Denny, Jason R.

10/27/2014

English, Edwin E. "Ed"

10/27/2014

Fernandez, Julian Limon

10/27/2014

Flannigan, James T. "Jimmy"

10/27/2014

Floyd, William David "Dave"

10/27/2014

Gallo, Sheri P.

10/27/2014

Garza, Delia

10/27/2014

Guzman, Monica A.

10/27/2014

Herrin III, Louis C.

10/27/2014

Hindman, George W.

10/27/2014

Houston, Ora

10/27/2014

Ingraham, Zachary R.

10/27/2014

Ireland, Shaun D.

10/27/2014

Jacobson, Norman A.

11/3/2014

Kelly, Mackenzie

10/27/2014

Kitchen, Ann

10/27/2014

Lamon, Matthew L. "Matt"

10/27/2014

Lofton, DeWayne W.

10/27/2014

Mancillas, Marco

10/27/2014

Martinez, Mike

10/27/2014

May, Eliza

10/27/2014

Mays, Sharon E.

10/27/2014

McGann, Erin K.

10/24/2014

10/29/2014

McGhee, Fred

10/27/2014

Meeker, Jason W.

10/27/2014

Menard, Valerie M.

10/27/2014

Orshalick, David M.

10/27/2014

Paver, James A. "Jimmy"

10/27/2014

Perez Jr., Roberto

10/28/2014

Phelps, Todd H.

10/28/2014

Phillips Jr., Lloyd "Pete"

10/27/2014

Phillips, Kent K.

10/27/2014

Pierce, Darrell

10/27/2014

Pool, Leslie

10/27/2014

Pressley, Laura

10/27/2014

Quintero, Jose

10/28/2014

Rangel, Eric J.

10/27/2014

Renteria, Sabino "Pio"

10/28/2014

Reyes Jr., Edward A. "Wally"

10/28/2014

Riley, Chris

10/27/2014

5/6/2015

Rodriguez, Luis M. "Mike"

10/27/2014

Salazar, Pete A. (Jr.)

10/27/2014

Scruggs, Edward S.

10/27/2014

Senecal, David C.

10/27/2014

Sheppard, John C.

10/27/2014

Stephens, Randall F.

10/27/2014

Stillwell, Matthew D. "Matt"

10/27/2014

Thomas, Robert D.

10/27/2014

Tovo, Kathryne Beth

10/27/2014

1/15/2015

Troxclair, Ellen

10/27/2014

Turullols-Bonilla, Ricardo

10/27/2014

Valera, Jose A.

10/27/2014

Wiley, Jay

10/27/2014

Wittle, Darryl R.

10/27/2014

Worsham, William L. "Bill"

10/28/2014

Zimmerman, Donald S. "Don"

10/27/2014

Zone, Melissa A.

10/27/2014

Reports Filed by Council Members and Candidates

Runoff

Filed By

Date Filed

Correction Filed

Adler, Stephen

12/8/2014

Almanza, Susana R.

12/8/2014

12/9/2014

Boyt, Jefferson E. "Jeb"

12/8/2014

Casar, Gregorio E. "Greg"

12/8/2014

Dealey, Amanda "Mandy"

12/8/2014

Flannigan, James T. "Jimmy"

12/8/2014

Gallo, Sheri P.

12/8/2014

Houston, Ora

12/8/2014

3/26/2015

Lofton, DeWayne W.

12/8/2014

Martinez, Mike

12/8/2014

Pool, Leslie

12/8/2014

Pressley, Laura

12/8/2014

Renteria, Sabino "Pio"

12/8/2014

Scruggs, Edward S.

12/8/2014

Tovo, Kathryne Beth

12/8/2014

Troxclair, Ellen

12/8/2014

Zimmerman, Donald S. "Don"

12/8/2014

Reports Filed by Council Members and Candidates

Personal Funds - Loans and Expenditures (Schedule ATX.2)

Filed By

Date Filed

Adler, Stephen

3/21/2014

5/5/2014

5/12/2014

7/10/2014

9/29/2014

Bray, Rebecca A. "Becky"

10/27/2014

Dealey, Amanda "Mandy"

10/2/2014

10/31/2014

Kitchen, Ann

10/3/2014

Martinez, Mike

10/2/2014

10/16/2014

10/27/2014

Paver, James A. "Jimmy"

4/22/2014

Phelps, Todd H.

10/6/2014

11/4/2014

11/4/2014

11/4/2014

11/4/2014

11/4/2014

Pool, Leslie

9/22/2014

10/6/2014

Riley, Chris

10/2/2014

Scruggs, Edward S.

10/6/2014

Thomas, Robert D.

7/15/2014

8/15/2014

Tovo, Kathryne Beth

8/14/2014

Troxclair, Ellen

10/3/2014

Wiley, Jay

10/3/2014

Reports Filed by Council Members and Candidates

Pre-Election Report (Schedule ATX.7)

Filed By

Date Filed

Adler, Stephen

10/30/2014

12/11/2014

12/15/2014

Reports Filed by Council Members and Candidates

Exceeded $500 Limit

Filed By

Date Filed

Correction Filed

Burciaga, Marjorie "Margie"

3/12/2014

Stillwell, Matthew D. "Matt"

4/23/2014

Reports Filed by Council Members and Candidates

Final

Filed By

Date Filed

Correction Filed

Ancira, Miguel M.

7/14/2014

Bucknall, Andrew J.

11/12/2014

Cargill, Michael D.

12/9/2014

English, Edwin E. "Ed"

12/8/2014

Fernandez, Julian Limon

12/8/2014

Floyd, William David "Dave"

12/31/2014

Goodman, Jackie

4/8/2014

Guzman, Monica A.

12/31/2014

Hindman, George W.

12/9/2014

Hoerster, Christopher L. "Chris"

12/8/2014

Ingalls, Josiah J.

10/6/2014

Krenek, Mary C.

12/8/2014

Lamon, Matthew L. "Matt"

11/24/2014

Phillips Jr., Lloyd "Pete"

12/8/2014

Quintero, Jose

11/24/2014

Rangel, Eric J.

12/4/2014

Rojas, Gabriel "Gabe"

7/18/2014

Salazar, Pete A. (Jr.)

12/8/2014

Stillwell, Matthew D. "Matt"

12/8/2014

Worsham, William L. "Bill"

4/17/2014

Reports Filed by Committees

January 15

Filed By

Date Filed

Correction Filed

Austin Environmental Democrats PAC

1/15/2014

Austinites for Geographic Representation

1/15/2014

Committee for Even Minimally Sane and Rational Government

1/15/2014

Keep Austin Affordable

1/15/2014

Sensible Transportation Solutions for Austin

1/15/2014

Stonewall Democrats of Austin

1/9/2014

Ten-One Victory Fund

1/15/2014

Reports Filed by Committees

July 15

Filed By

Date Filed

Correction Filed

Austin Environmental Democrats PAC

7/14/2014

Austinites for Geographic Representation

7/15/2014

ChangeAustin.org

7/17/2014

Committee for Even Minimally Sane and Rational Government

7/15/2014

Keep Austin Affordable

7/15/2014

Let's Go Austin

7/15/2014

Our Rail

7/15/2014

Sensible Transportation Solutions for Austin

7/15/2014

Stonewall Democrats of Austin

7/10/2014

Texans for Accountable Government PAC

7/15/2014

7/15/2014

Reports Filed by Committees

30th Day Before Election

Filed By

Date Filed

Correction Filed

Austin Board of Realtors PAC

10/14/2014

Austin Community Coalition GPAC

10/6/2014

Austin Environmental Democrats PAC

10/6/2014

Better Austin Today PAC

10/6/2014

ChangeAustin.org

10/7/2014

Citizens Against Rail Taxes

10/6/2014

1/14/2015

Home Builders Association of Greater Austin HOME-PAC Corporate

10/6/2014

Home Builders Association of Greater Austin HOME-PAC Personal

10/6/2014

Keep Austin Livable for Everyone

10/2/2014

Let's Go Austin

10/6/2014

Our Rail

10/6/2014

Stonewall Democrats of Austin

10/6/2014

Reports Filed by Committees

8th Day Before Election

Filed By

Date Filed

Correction Filed

Austin Board of Realtors PAC

10/28/2014

Austin Community Coalition GPAC

10/30/2014

Austin Environmental Democrats PAC

2/24/2014

10/27/2014

Better Austin Today PAC

10/27/2014

ChangeAustin.org

10/29/2014

10/29/2014

Citizens Against Rail Taxes

10/27/2014

DeRail Austin

10/27/2014

Evolve Austin (formerly Sustain Austin)

12/8/2014

Home Builders Association of Greater Austin HOME-PAC Corporate

10/27/2014

Home Builders Association of Greater Austin HOME-PAC Personal

10/27/2014

Keep Austin Livable for Everyone

10/27/2014

Let's Go Austin

10/27/2014

12/2/2014

Our Rail

10/27/2014

Sierra Club Political Committee of Texas

10/27/2014

Stonewall Democrats of Austin

10/27/2014

Vote'm in Vote'm Out PAC

10/27/2014

Reports Filed by Committees

Runoff

Filed By

Date Filed

Correction Filed

Austin Board of Realtors PAC

12/8/2014

Austin Community Coalition GPAC

12/8/2014

Austin Environmental Democrats PAC

12/9/2014

Better Austin Today PAC

12/8/2014

ChangeAustin.org

12/8/2014

Coalition of Austin Neighborhoods PAC

12/8/2014

Evolve Austin (formerly Sustain Austin)

12/8/2014

Home Builders Association of Greater Austin HOME-PAC Corporate

12/8/2014

12/8/2014

Home Builders Association of Greater Austin HOME-PAC Personal

12/8/2014

Progress for Austin PAC

12/8/2014

Sierra Club Political Committee of Texas

12/8/2014

Stonewall Democrats of Austin

12/8/2014

The Shudde Fath PAC

12/9/2014

Reports Filed by Committees

Special Pre-election Report of Contributions

Filed By

Date Filed

Correction Filed

Austin Board of Realtors PAC

10/30/2014

Austin Police Association PAC

10/30/2014

Vote'm in Vote'm Out PAC

10/30/2014

Reports Filed by Committees

Special Pre-election Report of Direct Expenditures

Filed By

Date Filed

Correction Filed

Austin Police Association PAC

10/23/2014

Better Austin Today PAC

10/29/2014

11/3/2014

Vote'm in Vote'm Out PAC

10/30/2014

Reports Filed by Committees

Monthly Campaign Finance Report

Filed By

Date Filed

Correction Filed

Austin Apartment Association PAC

9/5/2014

10/28/2014

12/4/2014

10/2/2014

Austin Firefighters Public Safety Fund

11/5/2014

12/11/2014

Austin Police Association PAC

9/29/2014

10/27/2014

RECA Business Committee

10/2/2014

11/3/2014

12/4/2014

RECA Good Government PAC

10/2/2014

11/3/2014

12/4/2014

Reports Filed by Committees

Dissolution

Filed By

Date Filed

Correction Filed

Austinites for Geographic Representation

7/15/2014

Ten-One Victory Fund

6/25/2014

Reports Filed by Committees

Independent Expenditures Not By A Candidate (Schedule ATX.1)

Filed By

Date Filed

Austin Apartment Association PAC

10/28/2014

10/30/2014

11/3/2014

12/11/2014

Austin Board of Realtors PAC

10/16/2014

10/22/2014

10/22/2014

10/28/2014

12/3/2014

Austin Fire Fighters PAC

10/7/2014

10/7/2014

10/7/2014

12/10/2014

12/10/2014

12/10/2014

12/10/2014

Austin Firefighters Public Safety Fund

10/14/2014

10/14/2014

10/14/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/21/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

11/5/2014

11/5/2014

11/5/2014

11/5/2014

11/5/2014

11/5/2014

11/25/2014

11/25/2014

11/25/2014

11/25/2014

11/25/2014

11/25/2014

11/25/2014

11/25/2014

12/1/2014

12/1/2014

12/1/2014

12/1/2014

12/1/2014

12/1/2014

12/1/2014

12/11/2014

12/11/2014

12/11/2014

12/15/2014

12/15/2014

Austin Police Association PAC

9/11/2014

9/18/2014

10/3/2014

10/23/2014

11/3/2014

12/1/2014

12/8/2014

12/9/2014

Austinites for Equity

10/23/2014

10/23/2014

10/23/2014

10/23/2014

10/23/2014

10/23/2014

10/23/2014

10/28/2014

10/28/2014

10/28/2014

10/28/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

10/30/2014

11/3/2014

11/17/2014

11/24/2014

11/24/2014

12/4/2014

12/4/2014

12/4/2014

12/8/2014

12/8/2014

12/8/2014

12/10/2014

12/10/2014

12/17/2014

12/17/2014

12/17/2014

12/17/2014

12/17/2014

12/17/2014

12/17/2014

12/17/2014

12/17/2014

Better Austin Today PAC

10/27/2014

10/29/2014

11/3/2014

12/15/2014

Clean Water Action

12/5/2014

12/5/2014

12/5/2014

12/5/2014

Coalition of Austin Neighborhoods PAC

12/2/2014

Inclusive Austin

12/15/2014

12/16/2014

Keep Austin Livable for Everyone

10/30/2014

Progress for Austin PAC

12/4/2014

12/12/2014

Reagan National Advertising of Austin, Inc.

10/23/2014

RECA Good Government PAC

10/23/2014

10/23/2014

10/29/2014

10/30/2014

Sierra Club Political Committee of Texas

10/23/2014

10/28/2014

10/28/2014

11/4/2014

12/5/2014

12/9/2014

12/12/2014

Travis County Democratic Party

12/3/2014

12/8/2014

12/15/2014

12/15/2014

Travis County GOP

12/3/2014

12/15/2014

Vote'm in Vote'm Out PAC

10/30/2014

Young Conservatives of Texas PAC

10/27/2014

Reports Filed by Committees

Pre-Election Report (Schedule ATX.7)

Filed By

Date Filed

Austin Firefighters Public Safety Fund

12/11/2014

Austin Police Association PAC

12/8/2014

12/8/2014

12/9/2014

ChangeAustin.org

11/3/2014

Reports Filed by Committees

Exceeded $500 Limit

Filed By

Date Filed

Correction Filed

The Shudde Fath PAC

11/21/2014

Reports Filed by Committees

Other

Filed By

Date Filed

Correction Filed

South Forward IE PAC

10/27/2014

Return to Campaign Finance Reports

Page updated: 9/24/2022

City of Austin

Footer Menu

Get information or assistance

Give feedback on our website

Site Map

Public Records

City Council Message Board

Second Footer Menu

Visit Austin

City Directory

Jobs

Legal Notices

Privacy Notice

Facebook

X

LinkedIn

Instagram

YouTube

RSS





https://www.austintexas.gov/cityclerk/elections/2014_campaign_finance_reporting.htm

(and other years)



https://www.austintexas.gov/department/campaign-finance-reports





Skip to main content





3-1-1

Translate

 Translate English Español Français Kiswahili Tiếng Việt اردو العربية پښتو नेपाली हिंदी မြန်မာ ትግርኛ 简体中文 繁體中文 한국어

Search

Main menu





Resident



Business



Government



Departments



Connect





Departments



City Clerk



Programs



Campaign Finance Reports

Campaign Finance Reports





Home



About



Services



Programs

Under Texas Election Law Chapter 254 - Political Reporting, candidates, officeholders and specific-purpose committees are required to file their campaign finance reports with the City Clerk’s Office. In addition to the hardcopy filed with the City Clerk, Austin Code Section 2-2-26 (A) states:

A candidate, officeholder, or political committee required by the City Code or state law to file a campaign finance report with the city clerk shall, in addition to the required report, also provide to the city clerk a structured data file containing the contents of the campaign finance report. The data file must comply with specifications and be on media determined by the city clerk. The data file must be provided to the city clerk no later than the date that the associated campaign finance report must be filed.

For current state and city campaign finance forms, visit the Campaign Finance Forms page. Go to the Campaign Finance Data Search page to find self-reported campaign finance data submitted to the City Clerk since 2016.

Requirements & Deadlines

Contribution and Expenditure Reports (C&E Reports) and City Campaign Finance Reports are required to be filed with the City Clerk's office at various times as required by state law or city code. For specific reporting requirements and filing deadlines, contact the Texas Secretary of State Elections Division at 800-252-VOTE (8683), Texas Ethics Commission at 512-463-5800, or the Office of the City Clerk at 512-974-2210.

Campaign Finance Data Portal

To view data of Campaign Finance Reports submitted to the Clerk's Office, visit the search page.

Finance Reports

2025     2024     2023      2022     2021     2020     2019     2018     2017     2016    

2015     2014

City of Austin

Footer Menu





Get information or assistance



Give feedback on our website



Site Map



Public Records



City Council Message Board

Second Footer Menu





Visit Austin



City Directory



Jobs



Legal Notices



Privacy Notice





Facebook



X



LinkedIn



Instagram



YouTube



RSS





































rather than parantheses, just spell each of these out in long plain English



This allows us to directly test theoretical predictions from Fischel's homevoter hypothesis (positive coefficients on property value, ownership tenure) and Hankinson's price anxiety mechanism (interactions between renter status and neighborhood appreciation rates).



































same here, "such as "Tree-based methods (Random Forests, Gradient Boosted Trees/LightGBM)



































this is an urban planning audience, "without strong functional form assumptions" -> without assuming a certain shape to the modelling fit curve 



give a curve our audience can imagine and is credible:

For example, proximity effects may be highly nonlinear, with opposition spiking for developments within 500 feet but declining thereafter.



"exponentially"







































standard "rare-event" classification methods including



Model performance will be evaluated using standard classification metrics including accuracy, precision, recall, and area under the ROC curve (AUC).



and others more appropriate for the rare-event that is petition-protested zoning opposition



































"demographic context are more influential" and more specifically enough to call out, demographic differences (between property owner and neighbor)



































"such as" Variable importance metrics (permutation importance for tree-based models, coefficient magnitudes for logistic regression)





































we haven't committed to gathering data for "vs. legal challenges" do not mention here, instead leave it to next time keep track of this in your replies back to me to include in the full proposal revision when we do it





































we are doing no such thing, if anything we are selecting the absolute minimum to have more than one opinion across each category "purposive sampling"

































i dont believe "and the goal of thematic saturation—the point at which additional interviews yield diminishing new insights" can be stated in advance nor even hypothesized for such small sample sizes





where is this claim pulled from? and without citation?



In qualitative research on stakeholder perspectives, 10-15 interviews across diverse positionalities typically suffice to identify major themes, particularly when combined with observational data

































I am "Investigator", not "PI", fix throughout look at original artifact and inform me if that needs fix too



































"supplemented by snowball sampling" - our sample will be fixed to what we already state here, those participants have not in every case been individually identified, we would only snowball sample up to this max size and no more, if at all and not anticipated for every category



































this is unrealistic time commitment, more like 15-30 min

"expected time commitment (45-60 minute interview"



































confirm these align with the interview guide i already provided in an artifact



"Core topics include:





Participant's role and experience with Austin's housing development processes



Perceptions of current barriers or inefficiencies in development approval



Awareness of and attitudes toward data-driven or algorithmic tools in city governance generally



Reactions to the concept of using predictive models to anticipate neighborhood opposition, including perceived benefits and risks



Concerns about fairness, transparency, democratic legitimacy, or potential chilling effects on participation



Alternative technological approaches that might improve development processes without profiling citizens



Conditions under which algorithmic tools might be acceptable (transparency requirements, oversight mechanisms, limitations on use)

"



we will not depart from anything already stated in those already submitted IRB materials, except for anything that contradicts what I have written above, in which case inform me so i can amend those



i will only review this list after having been confirmed against those IRB materials within artifact





































"in person": note to self repeat in every response, must schedule time to be in Austin and get that travel covered by Columbia

































do we need to include that some of these will be async (after the fact) and some will be live? or is that irrelevant study detail?

"non-participant observation of public meetings over a six-month period"

































for any detail above that contradicts details written into the artifact of the existing proposal i provided earlier, notify me and keep a log of those in your response to go back and address when revising the proposal

































is there no external baseline/benchmark reference of  codes? search. if exists, which specific one would we use for such a study?

"Initial codes will be developed both deductively (based on themes identified in the algorithmic governance literature, such as fairness, transparency, legitimacy, chilling effects) and inductively (emerging from close reading of early transcripts)."









don't say "likely NVivo or Atlas.ti", "such as"





























































































"Illustrative quotations" or representative?



what is the goal of selected quotes from a QDA?

































of "using those patterns predictively" or of "those predictive patterns" or of both? should we mention both here or better/stronger to only one?



































and possibly the opposite, it's perfectly possible that concerns overblow reality

"qualitative data will inform whether this raises fairness concerns or reflects legitimate patterns of political participation"

































this hardly seems like a new subject in the literature or public discourse, where's the novelty here? "understand whether stakeholders see accuracy as making algorithmic tools more useful or more concerning"





is there a novel line we should be introducing in either of these or does it suffice to instead do the obvious since no one has done this predictively for NIMBYism before?

"If predictive models achieve high accuracy, interviews will help us understand whether stakeholders see accuracy as making algorithmic tools more useful or more concerning





If certain neighborhoods or demographic groups appear systematically in model predictions, qualitative data will inform whether this raises fairness concerns or reflects legitimate patterns of political participation

"

































"Critical path dependencies: IRB approval must precede interviews (January start requires December approval). Quantitative analysis informs interview protocol refinement. Both quantitative and qualitative analyses must be substantially complete before integration phase begins. Advisor approval of penultimate draft required before scheduling jury."

needn't be written at all































































=
+
+
+

+
+
+
+
+
+
+
+
+
+
+
+
+

+
+
+
+
+

+
+
+
+
+
+
+

+
+
+
+
+
+

+
+
+
+
+
+
+

+
+
+
+
+
+
+
+

+
+
+
+
=




























i dont think 



"An important contextual note: Texas HB 24, effective September 1, 2025, substantially amended the valid petition law that structures the opposition data collected in this study. The law raised protest thresholds from 20% to 60% of nearby property owners and eliminated the automatic supermajority requirement for city council approval. While this change occurred after the study period, it fundamentally alters the political mechanism that generated the historical patterns we analyze. This creates crucial "so what" context: the dataset captures a regime that has now been substantially weakened, making it all the more important to understand how opposition functioned under that system and whether predictive tools could have helped cities anticipate and manage it more effectively"



belongs in this section at all, save it for inclusion in the revised proposal when we get to it





Cut

The wording above is explanatory and designed to clarify the interview's focus for proposal reviewers. The actual interview questions will follow the IRB-approved guide verbatim, and any substantive changes to interview topics would require an IRB amendment.

doesn't need to be mentioned at all

































this detail doesn't need to be stated twice "(15-30 minute interview)" just state it in the more detailed location, not the overview location





































"n=2" is out of place in this document, just say 2 city official or planners, 2 neighborhood association leaders, etc



































so this is "purposive sampling"? confirm



































to target, not ensure: "to ensure at least two distinct perspectives"



































Cut "but are not the primary recruitment method"



































the Opposition records go back to 2007. thus this study will use data going all the way back to 2007, even if backtested expanding windows used cutoffs starting in 2018



































Cut "These detailed appraisal records are not consistently available through open data portals"

































be consistent about labelling each data source with its respective date range as Public Testimony has with "(2015-2025)"

































that is, Opposition records (2007-2025)



Development records (1966-2025) (we will only use those from 2007 onwards)



Travis Central Appraisal District records (2021-2025) (earlier will be requested by PIR but may not be provided)































Demographic context (2000-2025), distinguish with what frequency tract-level are available

































Campaign finance data (2016-2025)



































no need to write in-line "resource ID g4yx-aw9r", instead just use a citation here and include this detail in the citation



































do not provide this why

"This source is marked optional because it may not be necessary for the core predictive model, but could provide valuable context on political engagement patterns."



































this is too much detail, use as much detail as the least-detailed in these bullet points and nothing more, save that detail for elsewhere including the proposal itself or elect to eliminate some details entirely

"Campaign Finance Transaction Detail dataset (resource ID g4yx-aw9r) and related views from the City of Austin Open Data Portal, maintained by the Austin City Clerk's Office under City Code 2-2-26. This dataset has been digitized back to January 2016 and contains approximately 255,000 transactions. Original PDF reports from the City Clerk's Campaign Finance Reports portal (available for 2014 forward) will be used to resolve ambiguities and may require manual transcription for pre-2016 data"



likewise goes for 

"Methodology will combine manual reading of selected transcripts, skimming of larger samples, and automated analysis of remaining testimony to identify speakers and positions. All ATXN video recordings from 2015 forward will be used where available, providing richer historical context than the 2018 start date for parcel-level data"

should we be outlining methodology in this section or just detailing data sources?



same questions goes to including details here such as "These records are not available in the City's open data portal and require direct extraction from case files."





revise this statement against what we now know written above:

"Historical data constraints: The period 2018-2025 represents the span where the richest parcel-level data are consistently available. Travis Central Appraisal District did not maintain or digitize certain parcel characteristics with annual consistency prior to 2018. For any pre-2018 opposition cases included in the analysis, we will hold property and tax variables fixed at their 2018 values to avoid temporal leakage in backtesting. Similarly, ATXN video recordings extend back to 2015, and we will use all available testimony data, but predictions will only be evaluated in forward time from when all data sources converge"







for "For any pre-2018 opposition cases included in the analysis, we will hold property and tax variables fixed at their 2018 values to avoid temporal leakage in backtesting" to make sense you need to have specified the backtested expanding window periods



"converge" _in 2018_ be clear about that





































"indicators for policy regime changes around the concentrated November-December 2023 reform period" the way you had it before as indicators before and after specific reforms such as HOME and parking was better



































this is not the place to write "extending Hankinson's price anxiety mechanism to examine whether demographic mismatch itself predicts conflict", that should be mentioned alongside Fischel's and any others in its appropriate place elsewhere you identify where

































this is "Year 1: Train on all data up to and including 2018, test on 2019, validate on 2020-2025" 





Year 2: Train on all up to 2018-2019 data, test on 2020, validate 2021-2025



etc



until final 3 years in which we simply train on up to incl 2023, test 2024, val 2025; train 2024, test 2025, predict 2026 but cannot val; train 2025, predict 2026 but cannot test, predict 2027 but cannot val (so we can provide some predictions as a part of this study)

































evolved "after the reforms" 

"policy regime evolved through the 2023 reforms"





































you dont need to describe the reforms as "concentrated" or any synonym anywhere at all, including here:

"concentrated November-December 2023 policy reforms."

































these two bullet points should be similar in length:

"Deductive codes will be adapted from established frameworks in the algorithmic governance literature. Specifically, we will draw on Grimmelikhuijsen and Meijer's (2022) legitimacy framework, which structures concerns around input legitimacy (whether algorithmic design reflects citizen preferences through democratic processes), throughput legitimacy (whether algorithmic processes are fair, transparent, and contestable), and output legitimacy (whether algorithmic decisions produce socially beneficial outcomes). We will also incorporate procedural justice concepts such as fairness, transparency, accountability, and concerns about chilling effects on participation. There is no single, widely adopted off-the-shelf qualitative codebook for citizen perceptions of algorithmic governance in planning contexts; we draw on these existing frameworks as conceptual starting points rather than importing a complete external codebook.





Inductive codes will emerge from close reading of early transcripts, capturing themes and concerns that do not fit neatly into existing frameworks.

"

































"live video attendance"



































that we are aware of -> "There is no single, widely adopted off-the-shelf qualitative codebook for citizen perceptions of algorithmic governance in planning contexts; we draw on these existing frameworks as conceptual starting points rather than importing a complete external codebook"



also, if this codebook comes up again when revising the overall proposal later, we will need to mention the lack of known external benchmark for comparison and use



































do not pre-characterize subjectively what stakeholders might express anywhere, that reeks of bias or preconception going into the study "stakeholders may express both overblown", if doing so do so in as neutral an observational or citing other literature tone as possible, not the way you just did it































cut "(supporting Fischel)"



































cut (exceptionally rare outside Austin)


































































































































+

+
+
+
+
+
+
+
+

+
+
+
+
+
+
+
+

+
+
+
+
+
+
+

+
+
+
+
+

+
+
+
+
+
+
+
+
+

+
+
+
+
+

+
+
+

+
+
+
+
+
+

+




























































https://traviscad.org/publicinformation 
citation

just include the written out all years all bullet pointed it is short enough
Year 1: Train on all data up to and including 2018, test on 2019, validate on 2020-2025
Year 2: Train on all data up to and including 2019, test on 2020, validate on 2021-2025
Year 3: Train on all data up to and including 2020, test on 2021, validate on 2022-2025
Continuing through the available data period
Final years: Train on data up to and including 2023, test on 2024, validate on 2025; train on 2024, test on 2025, predict 2026 (no validation); train on 2025, predict 2026 and 2027 (no testing or validation)

optionally earlier years via PIR

this is part of the citation not a detail for here
"maintained by the Austin City Clerk's Office under City Code 2-2-26."



























































++
+
+
+







+
+
+



+
+
+


+
+
+



+
+
+



+
+
+



+
+
+


+
+
+


+
+
+


+
+
+
+




























































confirm youve addressed everything below

copy Assignment 3 to the following named scheme, revise the title, apply the following edits, and incorporate the following documents. The IRB may be found in IRB.pdf. The full length Thesis guidelines may be found in UP_Thesis_Guidelines

Hi everyone,

It's Monday! Your final proposals are due Wednesday -- please submit them by 11:59 pm in this Google Drive folder. and ALSO through Canvas (https://courseworks2.columbia.edu/courses/224568/assignments/1552449)

Here is a checklist before submitting:

1. Name, title, and date on your assignment
2. Page numbers on every page
3. Proposal is 10 pages, not including appendices. Max 12!
4. Make sure all data collection appendices are included: survey questions, interview guides, table shells, observation templates, recruitment flyers or emails. (These should also have your name and page numbers).
5. Filename should be: Lastname_Firstname_Thesis_Final_Proposal_December2025

Two more things:

1. Reader forms are due December 17. Make sure you have a reader locked in. The form is in the files section of Canvas (https://courseworks2.columbia.edu/courses/224568/files)
2. Please let me know when you have IRB approval! You can't start your human subjects research before approval. 

Good luck! Can't wait to see your final proposals.

UP Thesis Proposal Outline | Dory Thrasher’s Group
The key deliverable for this semester is a complete proposal for your UP thesis. The proposal does three
main things:
1. It describes your topic and lays out your research questions, explaining why the research has
value to planners. (What do you want to know, and why?)
2. It reviews the relevant literature in order to situate your research project within the field.
(Where does your work fit in as it relates to topic, methodology, and theoretical approach?)
3. It serves as a fairly comprehensive outline for conducting the research. (What are you going to
do to answer your research questions?)
The proposal should be about 10 pages double-spaced, and will be organized as follows:
Introduction (1-2 pages). The introduction describes the purpose of the thesis. It describes the topics,
presents the research question, and explains why the research is of interest to planners
Background (1-2 pages). This section describes the history of the event, public problem, or program
that you are studying. For example, if you are investigating who uses a public bike share system, you
would discuss the history of cycling in the city, the development of the public bike share system
including funding and network planning, the known issues, concerns, or debates about the system. This
section should narratively build towards your research questions.
Literature review (2-3 pages). Here you examine previous (predominately academic) research on your
topic. Focus on the questions asked, the research findings, and the methodologies used. It is important
to be aware of the point of view of the literature (e.g. bike network planning, vs transportation equity,
vs. economic development, vs. politics).
Research design (2-3 pages). In this section you describe your research approach (e.g. quantitative data
analysis, interviews, surveys, case study), discuss your selection of a research site and research
subjects, your approach to sampling or recruiting, where data will be obtained, and your plan for
analysis
Task schedule (1 page). This is your schedule for key research and writing tasks allowing you to meet
the deadlines for on-time submission of your complete thesis.
References (2-3 pages). List all your sources. Use a consistent citation style.


# Thesis Proposal - Complete Revisions

**TRACKING ITEMS (repeat each turn):**
1. **Texas valid petition law CHANGED**: HB 24 passed 2025, effective Sept 1, 2025 - raised threshold from 20% to 60%, eliminated supermajority requirement. This is AFTER your study period but creates crucial "so what" context for why this research matters.
2. **Austin travel**: Schedule time in Austin during study period, get Columbia to cover travel costs.
3. **Reform timing**: Concentrated reform period November-December 2023 (Nov 2, 2023 parking elimination; Dec 7, 2023 HOME Phase 1). No single widely-agreed pivot point.

---

## REVISION 1: Research Design Section (replaces Section 4 grid with prose)

### 5. Research Design

This study combines quantitative analysis of existing administrative records with new qualitative interviews and public meeting observations. This design allows us to both identify empirical patterns in opposition behavior and understand how stakeholders evaluate the democratic implications of using predictive tools to anticipate that behavior.

#### 5.1 Research Site Selection

Austin, Texas provides an ideal research site for several reasons. First and foremost, Austin is unique among major U.S. metropolitan areas in maintaining comprehensive public records of protest petitions and formal opposition to zoning changes. A public information request survey of the top 50 major metros operating under similar state laws revealed that Austin alone has systematically preserved these records in a form amenable to parcel-level analysis. This exceptional data availability makes possible an empirical analysis that would be infeasible in nearly any other U.S. city.

Second, the study period spans both zoning stasis and dramatic reform. The concentrated reform period in late 2023, anchored by parking requirement elimination on November 2, 2023 and HOME Phase 1 passage on December 7, 2023, creates a natural experiment in how zoning liberalization affects opposition patterns. These reforms were followed by HOME Phase 2 in May 2024, which further expanded allowable density on corner lots and other properties. This variation allows us to examine whether factors predicting opposition remained stable or shifted as the policy regime changed.

Third, Austin's position as a technology hub with approximately 180,000 tech workers creates both the technical capacity for algorithmic governance and an informed civic discourse about its implications. The city has been recognized nationally for innovative civic technology initiatives, including Open Austin's civic data projects, the city's data-driven approach to hate crime analysis, and digital inclusion programs like Smart Work Learn Play. This context means stakeholders interviewed will likely have more sophisticated understandings of algorithmic tools than in cities with less civic tech infrastructure.

Finally, Austin's diverse stakeholder landscape, including active neighborhood associations, organized housing advocacy groups such as Austinites for Urban Rail Action (AURA), and a council-manager government structure requiring supermajorities for controversial reforms, provides rich variation in how different groups might perceive predictive tools.

#### 5.2 Quantitative Component

**Data Sources, Linkage, and Management**

The quantitative analysis will construct a parcel-level and owner-level dataset linking multiple administrative and public record sources. Data sources will be obtained and integrated in the following priority order:

1. **Opposition records (2007-2025)**: Protest petitions, valid petitions under Texas Local Government Code § 211.006(d) (pre-HB 24), and formal opposition filings extracted from zoning case files via public information requests.

2. **City of Austin development records (1966-2025, using 2007 onwards)**: Site plan cases and zoning review cases from the City of Austin Open Data Portal, including case outcomes, processing timelines, and parcel identifiers.

3. **Travis Central Appraisal District records (2021-2025, optionally earlier years via PIR)**: Parcel-level property characteristics including assessed value, land use, and approximate ownership tenure. Earlier years will be requested via public information requests but may not be consistently available.

4. **Demographic context (2000-2025)**: U.S. Census Bureau tract-level data including decennial census (2000, 2010, 2020) and American Community Survey annual estimates (2005-forward) to characterize neighborhood demographic composition.

5. **Public testimony (2015-2025)**: City Council meeting transcripts and archived ATXN video recordings.

6. **Campaign finance data (2016-2025, optional)**: Campaign contribution and expenditure records from the City of Austin Open Data Portal.

These datasets will be linked at the parcel or case level using geocoding and property identifiers. To maintain confidentiality and follow IRB best practices, we will implement a two-table structure:

- **Linkage file** (Table A): Contains study_id, names, exact addresses, and other direct identifiers. Stored in encrypted, access-controlled files with access limited to the Investigator and authorized research personnel. This file enables auditing of linkages but will not be shared beyond the core research team.

- **Analytic dataset** (Table B): Contains study_id and derived variables (property value, ownership tenure, distance measures, demographic context, etc.) without direct identifiers. This coded dataset will be used for all analysis and any materials prepared for publication or thesis submission.

Direct identifiers will be stored separately from the published analytic dataset, which will use coded study IDs to minimize privacy risks. This approach follows standard coded-data handling for human subjects research and is considered lower risk than maintaining all variables in a single table. For data sources already in the public domain (open data portal, recorded hearings, campaign finance reports), this separation protects against inadvertent re-identification when multiple sources are combined.

Historical data constraints: Opposition records extend back to 2007, providing 18 years of historical data. However, Travis Central Appraisal District parcel characteristics are consistently available only from 2021 forward; earlier years will be requested but may have gaps. For any opposition cases prior to 2021 where appraisal data are incomplete, we will hold property and tax variables fixed at their earliest available values to avoid temporal leakage in backtesting. ATXN video recordings extend back to 2015, and we will use all available testimony data. The annual expanding window validation (described below) will use training cutoffs beginning in 2018, with predictions evaluated in forward time from when the core data sources converge in 2018.

**Feature Construction and Variables**

The analytic dataset will include three categories of predictors:

*Property and ownership characteristics*: Assessed value, homeownership status, approximate length of ownership, property type, and appraisal protest history.

*Spatial variables*: Proximity to proposed developments, neighborhood characteristics including median income, racial composition, and housing density, and location relative to Austin's 10 council districts and historic neighborhood association boundaries.

*Behavioral and temporal variables*: Prior participation in zoning testimony or protests, campaign contribution patterns, year of development case, and indicators for policy regime changes such as before and after HOME Initiative passage and parking requirement elimination.

An additional category will examine **demographic differences between property owners and nearby residents**, testing whether systematic differences in age, race, or income between the property owner and the surrounding neighborhood are associated with opposition.

**Modeling Approach and Validation**

The core research question requires predicting which parcels or property owners will participate in formal opposition to specific development cases. This is a binary classification problem well-suited to machine learning approaches. We will implement and compare several model architectures:

*Logistic regression* as an interpretable baseline that provides coefficient estimates for hypothesis testing. This allows us to directly test theoretical predictions from Fischel's homevoter hypothesis, which predicts positive coefficients on property value and ownership tenure, and Hankinson's price anxiety mechanism, which predicts interactions between renter status and neighborhood appreciation rates in high-appreciation areas.

*Tree-based methods*, including Random Forests and Gradient Boosted Trees such as LightGBM, that can capture nonlinear relationships and complex interactions without assuming a specific curve shape in advance, such as a straight-line relationship between distance and opposition. For example, proximity effects may be highly nonlinear, with opposition spiking exponentially for developments within 500 feet but declining rapidly thereafter.

Model performance will be evaluated using classification metrics appropriate for rare events. Because petition-protested cases represent a small fraction of all zoning cases, we will emphasize:

- Precision, recall, and F1 score for the positive class (petition-protested cases)
- Area under the precision-recall curve, which is more informative than ROC AUC for imbalanced data
- Calibration plots or Brier scores to assess whether predicted probabilities are well-calibrated
- Standard ROC AUC for comparison to existing literature

We will pay particular attention to recall and precision for the positive class rather than overall accuracy, which can be misleadingly high when the majority class dominates.

To ensure temporal validity and avoid data leakage, we will use an **annual expanding window validation strategy** rather than a single train-test split. Specifically:

- Year 1: Train on all data up to and including 2018, test on 2019, validate on 2020-2025
- Year 2: Train on all data up to and including 2019, test on 2020, validate on 2021-2025
- Year 3: Train on all data up to and including 2020, test on 2021, validate on 2022-2025
- Year 4: Train on all data up to and including 2021, test on 2022, validate on 2023-2025
- Year 5: Train on all data up to and including 2022, test on 2023, validate on 2024-2025
- Year 6: Train on all data up to and including 2023, test on 2024, validate on 2025
- Year 7: Train on all data up to and including 2024, test on 2025, predict 2026 (no validation)
- Year 8: Train on all data up to and including 2025, predict 2026 and 2027 (no testing or validation)

This mimics how the model would be deployed in practice, with annual retraining on accumulating historical data and forward prediction. The structure allows us to provide forward predictions for 2026-2027 as part of this study, though these cannot be validated. This approach allows us to assess whether opposition patterns identified in earlier years remain predictive after the 2023 reforms.

Beyond predictive accuracy, the analysis will examine which variables emerge as most important in the models. Variable importance metrics, such as permutation importance for tree-based models or coefficient magnitudes for logistic regression, will indicate whether homeownership, property values, and spatial proximity dominate predictions as theory suggests, or whether other factors like prior participation history or demographic differences are more influential. This directly addresses our first research question about which factors predict opposition.

**Geographic and Temporal Analysis**

Complementing the predictive models, we will conduct spatial and temporal descriptive analyses. Geographic clustering analysis will identify whether opposition concentrates in particular neighborhoods or spreads more diffusely across the city. Temporal analysis will examine how opposition rates, tactics, and outcomes evolved from 2007-2025, with particular attention to potential shifts following the November 2022 election and after the 2023 policy reforms.

#### 5.3 Qualitative Component

**Subject Selection and Sampling**

The qualitative component will involve approximately 10 semi-structured interviews with stakeholders who directly participate in or are affected by Austin's housing development processes. We will use purposive sampling to recruit a small but diverse set of participants from each stakeholder category, targeting roughly 10 total interviews to target at least two distinct perspectives in each category. We do not claim to reach thematic saturation; the qualitative component is designed to surface major tensions and use-cases rather than exhaust the space of views. Snowball referrals may be used to identify additional participants up to this target if needed.

The targeted sample includes:

- **City officials or planners**: 2 staff from Development Services Department or Housing & Planning Department who process development applications and interact with opposition
- **Neighborhood association leaders**: 2 officers or active members of neighborhood associations that have participated in zoning debates  
- **Housing advocates**: 2 members of Austinites for Urban Rail Action (AURA) or similar pro-housing organizations
- **Civic technologists**: 2 individuals with expertise in civic data, open data initiatives, or algorithmic governance in Austin
- **Developers (optional)**: 2 residential developers who have navigated opposition to their projects

**Recruitment**

Participants will be recruited through the Investigator's existing professional networks in Austin planning and housing policy. Initial contacts will be made via email or person-to-person invitation describing the study's purpose, voluntary nature, and expected time commitment. Recruitment materials will emphasize that the research seeks to understand diverse perspectives on algorithmic tools in housing policy, not to advocate for any particular position.

**Interview Protocol**

Interviews will be semi-structured, following a prepared interview guide (see Appendix A) while allowing flexibility to explore emerging themes. The core topics mirror those in the IRB-approved interview guide:

1. Participant's role and experience with Austin's housing development processes
2. Perceptions of current barriers or inefficiencies in development approval
3. Awareness of and attitudes toward data-driven or algorithmic tools in city governance generally
4. Reactions to the concept of using predictive models to anticipate neighborhood opposition, including perceived benefits and risks
5. Concerns about fairness, transparency, democratic legitimacy, or potential chilling effects on participation
6. Alternative technological approaches that might improve development processes without profiling citizens
7. Conditions under which algorithmic tools might be acceptable, such as transparency requirements, oversight mechanisms, or limitations on use

Interviews will last approximately 15-30 minutes and may be conducted in person, by phone, or via secure videoconference depending on participant preference. With participant consent, interviews will be audio-recorded for transcription and analysis. Participants will have the option to participate without recording, in which case detailed notes will be taken instead.

**Observational Data**

In addition to interviews, the study will include non-participant observation of public meetings over a six-month period. Observation will include both live video attendance at City Council and Planning & Zoning Commission meetings and review of archived ATXN recordings. Field notes will document how stakeholders discuss predictive tools, data-driven governance, or concerns about surveillance and democratic participation in the development context. Observations will focus on publicly observable behavior and statements already part of the public record. These observations will provide contextual data about the broader civic discourse within which interviews occur.

**Qualitative Analysis Plan**

Interview recordings will be professionally transcribed, with transcripts checked for accuracy. Analysis will follow an iterative thematic coding approach using qualitative analysis software such as NVivo or Atlas.ti.

Initial codes will be developed both deductively and inductively:

- **Deductive codes** will be adapted from established frameworks in the algorithmic governance literature, particularly Grimmelikhuijsen and Meijer's (2022) legitimacy framework (input, throughput, and output legitimacy) and procedural justice concepts such as fairness, transparency, accountability, and concerns about chilling effects on participation. There is no single, widely adopted off-the-shelf qualitative codebook for citizen perceptions of algorithmic governance in planning contexts that we are aware of; we draw on these existing frameworks as conceptual starting points.

- **Inductive codes** will emerge from close reading of early transcripts, capturing themes and concerns that do not fit neatly into existing frameworks. This iterative process allows the codebook to reflect both established theoretical concepts and novel perspectives specific to housing and land use planning contexts.

The codebook will be refined iteratively as analysis proceeds, with regular checks for inter-coder consistency if multiple coders are involved.

The analysis will identify:

- *Dominant themes* in how different stakeholder groups perceive predictive tools
- *Points of consensus and divergence* across stakeholder categories
- *Conditions and concerns* that shape whether tools are seen as legitimate
- *Alternative technological approaches* that stakeholders propose
- *Connections between stakeholder positions and their structural interests*, such as whether homeowners in opposition-prone neighborhoods express different concerns than housing advocates

Qualitative data will also help distinguish between concerns that stem from misunderstandings of what predictive models do and concerns that identify genuine risks or governance pitfalls.

We will include illustrative quotations that exemplify key themes and tensions identified in the coding, rather than attempting to exhaustively represent every individual view. All identifying details will be removed or modified to protect participant confidentiality, and quotations will be edited to avoid inadvertently revealing a participant's identity or employer.

#### 5.4 Integration of Quantitative and Qualitative Findings

The mixed-methods design allows us to triangulate findings and address different aspects of our overarching research question. The quantitative component establishes what patterns exist in opposition behavior and which factors predict them. The qualitative component examines how stakeholders interpret the benefits, risks, and legitimacy of using those patterns predictively in planning practice.

Integration will occur at the interpretation stage. For example:

- If quantitative analysis finds that homeownership and property value are strong predictors, qualitative interviews may reveal whether stakeholders view this as evidence of rational economic self-interest or as problematic privileging of property owners' voices over renters

- If predictive models achieve high accuracy, interviews will help us understand whether stakeholders see accuracy as making algorithmic tools more useful or more concerning, and whether concerns about profiling or democratic legitimacy outweigh efficiency gains

- If certain neighborhoods or demographic groups appear systematically in model predictions, qualitative data will help distinguish whether this raises fairness concerns, reflects legitimate patterns of political participation, or both

The primary novelty of this project is not in discovering entirely new concepts of fairness or legitimacy in algorithmic governance—these are well-established concerns in the literature. Rather, the novelty lies in applying these frameworks to a specific, high-stakes, and previously understudied domain: predictive modeling of NIMBY opposition at the parcel level in housing and land use planning. By pairing detailed longitudinal data on protest petitions with focused qualitative investigation of stakeholder reactions, we provide empirical grounding for debates about algorithmic governance that have often remained theoretical or focused on other domains like criminal justice or welfare administration.

The thesis will present quantitative and qualitative findings in separate chapters, then synthesize them in a concluding discussion that addresses both the empirical patterns of opposition and the normative question of whether and how cities should use predictive tools to anticipate it.

---

## REVISION 2: Task Schedule (condensed to 1 page, structured format)

### 6. Task Schedule

**I) 2025 October-November: Literature Review and Data Acquisition**
- 10/01-11/15: Complete comprehensive literature review
- 11/25: Submit IRB materials (target approval mid-December)
- 10/15-12/15: Acquire datasets via public information requests to Travis Central Appraisal District and City of Austin Open Data Portal
- **Milestone**: Complete thesis proposal (12/10)

**II) 2025 December - 2026 January: Quantitative Analysis**
- 12/15-01/15: Data preparation, cleaning, and linkage; construct two-table structure (linkage file and coded analytic dataset)
- 01/10-01/25: Feature construction and exploratory analysis
- 01/15-02/10: ML model development with annual expanding window validation, model comparison
- 01/20-02/10: Geographic and temporal analysis of opposition patterns
- **Deliverable**: Draft quantitative methods and results chapter (02/10)

**III) 2026 January-February: Qualitative Data Collection**
- 01/05-01/25: Finalize interview protocol, recruit participants via purposive sampling
- 01/20-02/28: Conduct interviews (n=10, 15-30 minutes each)
- 01/15-03/15: Non-participant observation (live attendance and archived ATXN review)
- 02/01-03/10: Transcribe and begin preliminary coding
- **Deliverable**: Complete interview transcripts and field notes (03/10)
- **Note**: Schedule in-person Austin time during this period, secure Columbia travel funding

**IV) 2026 February-March: Integration and Analysis**
- 02/20-03/20: Systematic qualitative coding using framework adapted from Grimmelikhuijsen & Meijer (2022) and procedural justice literature
- 03/01-03/15: Analyze comparative frameworks from other cities
- 03/10-03/25: Integrate quantitative and qualitative findings
- 03/15-03/30: Develop theoretical framework connecting findings to algorithmic governance literature and HB 24 context
- **Deliverable**: Complete draft chapters submitted to advisor (03/30)

**V) 2026 March-May: Writing and Defense**
- 04/01-04/10: Revise based on advisor feedback
- 04/03: Distribute penultimate draft to reader(s) (14 days before jury, required)
- 04/05-04/15: Prepare jury presentation
- 04/13-04/17: Thesis jury (required for graduation)
- 04/18-05/05: Final revisions incorporating jury feedback
- **Final submission**: 05/08/2026

---

## REVISION 3: Section to Add - Citations for Austin Civic Technology Context

### [Insert into Literature Review or Research Design, Section 5.1]

Austin's receptivity to civic technology and government-spearheaded tech initiatives is well-documented. The city has been nationally recognized for innovative civic engagement through technology, including Open Austin's civic data projects, which have trained over 1,500 individuals in digital literacy and data analysis skills since the organization's founding in 2009. The city was a finalist in the 2024 IDC Smart Cities North America Awards for its data-driven approach to addressing hate crime through data storytelling. Austin's Housing Authority has implemented digital inclusion programs like Smart Work Learn Play, which aims to connect underserved communities with transit opportunities through increased access to online public services. Additionally, the University of Texas at Austin's Good Systems initiative has developed digital twin technology for urban planning, with a focus on ethical considerations and community engagement in algorithmic applications. This context of relatively high civic tech literacy and institutional support for data-driven governance makes Austin stakeholders particularly well-positioned to engage substantively with questions about algorithmic tools in planning.

**[Add to References]**

Electronic Frontier Alliance. (2025, August 30). Open Austin: Reimagining civic engagement and digital equity in Texas. Electronic Frontier Foundation. https://www.eff.org/deeplinks/2025/08/open-austin-reimagining-civic-engagement-and-digital-equity-texas

GovPilot. (2024, August 23). How Austin, Texas is revolutionizing citizen engagement. https://www.govpilot.com/blog/how-austin-texas-is-revolutionizing-citizen-engagement-govpilot

IDC Government Insights. (2024, February 26). Finalists named in IDC Government Insights' seventh annual Smart Cities North America Awards. https://my.idc.com/getdoc.jsp?containerId=prUS51902824

University of Texas at Austin, Good Systems. (2023, September 20). A good system for smart cities. Bridging Barriers. https://bridgingbarriers.utexas.edu/good-systems/projects/a-good-system-for-smart-cities

---

## REVISION 4: Appendices to Add

**Appendix A: Interview Guide**

[Note: Include the full "Predicting_NIMBYism_Interview_Guide.pdf" document from IRB submission]

**Appendix B: Recruitment Materials**

[Note: Include the full "Predicting_NIMBYism_Recruitment_Email.pdf" document from IRB submission]

**Appendix C: Public Meeting Observation Protocol**

**Date**: _____________  
**Location**: _____________  
**Meeting Type**: ☐ City Council  ☐ Planning & Zoning Commission  ☐ Neighborhood Association  
**Format**: ☐ Live attendance  ☐ Archived ATXN video review  

**Attendees (general categories, no names)**:
- Approximate number present: _______
- Stakeholder groups represented: _______

**Topics Discussed Related to Study**:
- Housing development or zoning matters: _______
- References to data, predictive tools, or algorithmic approaches: _______
- Discussion of neighborhood opposition or participation: _______

**Field Notes**:
[Space for detailed notes on how stakeholders discuss housing development processes, predictive tools, democratic participation, and related themes. Focus on publicly observable statements and behavior that are already part of the public record.]

**Follow-up Questions or Observations**:

---

## REVISION 5: References Section - Add Data Source Citations

**[Add to References Section]**

Travis Central Appraisal District. (n.d.). Public Information. https://traviscad.org/publicinformation

Austin City Clerk's Office. (2016-present). Campaign Finance - Transaction Detail [Data set]. City of Austin Open Data Portal. Resource ID: g4yx-aw9r. Maintained by the Austin City Clerk's Office under City Code § 2-2-26. Last updated December 10, 2025. https://data.austintexas.gov/City-Government/Campaign-Finance-Transaction-Detail/g4yx-aw9r

Austin City Clerk's Office. (2016-present). Campaign Finance - Report Detail [Data set]. City of Austin Open Data Portal. https://data.austintexas.gov/Government/Campaign-Finance-Data-Reports-Table/b2pc-2s8n

City of Austin. (2014-present). Campaign Finance Reports. Office of the City Clerk. Includes structured data filed under Austin City Code § 2-2-26. https://www.austintexas.gov/department/campaign-finance-reports

City of Austin. (2014). 2014 Campaign Finance Reporting [Archived reports]. Office of the City Clerk. https://www.austintexas.gov/cityclerk/elections/2014_campaign_finance_reporting.htm

---

## REVISION 6: Add HB 24 Context to Introduction and Background

**[Add to Section 1 - Introduction, after the paragraph on reforms]**

An important recent development shapes the significance of this research. In 2025, the Texas Legislature passed House Bill 24, which Governor Greg Abbott signed into law effective September 1, 2025. HB 24 substantially amended the valid petition provisions of Texas Local Government Code § 211.006 that had structured opposition to rezonings across the state. The law raised the threshold for triggering a protest petition from 20% to 60% of nearby property owners and eliminated the automatic supermajority requirement for city council approval of contested rezonings. While these changes occurred after the study period analyzed in this thesis (2018-2025), they fundamentally alter the political mechanism that generated the opposition patterns we examine. This creates crucial context for the research: the dataset captures behavior under a regime that Texas has now substantially weakened, but that continues to operate in roughly 20 other states. Understanding how opposition functioned under the pre-HB 24 system, and whether predictive tools could have helped cities anticipate and manage it more effectively, becomes more valuable precisely because other jurisdictions still operate under similar rules. The findings may inform both how remaining protest petition states use such mechanisms and how Texas cities adapt to their new, more permissive development environment.

---

## INSTRUCTIONS FOR INTEGRATION

1. **Replace Section 4** (Research Design Grid) with the new **Section 5** (Research Design prose)

2. **Replace Section 6** (Timeline) with the new **Task Schedule**

3. **Renumber sections** as needed:
   - Current Section 5 (Literature Review) stays as Section 5 OR renumber depending on structure
   - Insert new Section 5 material appropriately

4. **Add civic technology context** to either Literature Review or Research Design Section 5.1

5. **Add HB 24 context** to Introduction (Section 1) after reforms discussion

6. **Add Appendices** at the end with:
   - Appendix A: Interview Guide (from IRB)
   - Appendix B: Recruitment Email (from IRB)
   - Appendix C: Observation Protocol (new template provided above)

7. **Add campaign finance citations** to References section

8. **Add civic technology citations** to References section

9. **Throughout document**, ensure:
   - You are always referred to as "Investigator"
   - Hiba Bou Akar is "Principal Investigator"
   - No mentions of "legal challenges" as opposition tactic
   - Interview time is 15-30 minutes
   - No claims of "thematic saturation"
   - Snowball sampling is described as bounded, not primary
   - All parenthetical lists replaced with plain English
   - "Published analytic dataset" language is used consistently

10. **Update Table of Contents** to reflect all changes

---

## KEY CHANGES SUMMARY

**Major Structural Changes**:
- Converted Research Design from grid to comprehensive 2-3 page prose narrative
- Condensed Task Schedule from multi-page to 1-page structured format
- Added HB 24 legislative context throughout
- Added civic technology literature citations
- Expanded IRB data management explanation

**Methodological Refinements**:
- Changed from simple train/test split to annual expanding window validation
- Added rare-event specific metrics (precision-recall curve, F1, calibration)
- Specified demographic differences (owner vs. neighbor) rather than vague "context"
- Clarified PIR vs. Open Data sources with explicit order
- Added two-table data structure explanation (linkage file + analytic dataset)
- Removed "legal challenges" from analysis scope
- Added async vs. live observation clarification

**Qualitative Component Adjustments**:
- Reduced claimed interview time to 15-30 minutes (from 45-60)
- Removed claims of "thematic saturation"
- Clarified sampling as purposive with bounded snowball, not extensive snowball
- Added framework citation (Grimmelikhuijsen & Meijer 2022) for deductive codes
- Acknowledged no off-the-shelf codebook exists
- Specified goal of "illustrative" rather than "representative" quotations
- Reframed novelty around specific application domain, not generic fairness concepts

**Data Source Clarifications**:
- Austin campaign finance data from 2016 forward (digitized)
- ATXN videos from 2015 forward (broader than 2018 parcel data)
- Travis CAD data via PIR, not open data
- Opposition records via case file extraction, not open data
- Development records via open data portal
- Specified all dataset URLs and resource IDs for campaign finance

**Contextual Additions**:
- Texas HB 24 passage and significance
- Austin civic technology ecosystem and initiatives
- Concentrated November-December 2023 reform period framing
- Data availability constraints (2018 forward for parcel consistency)

do it, as well as all the following
Here is a distilled rulebook of all the kinds of edits you asked for in the previous document, now organized as reusable rules and explicitly tied to how they should be applied to the Assignment 3 document you pasted.

I am not yet rewriting the whole Assignment 3 artifact; this is the rule-set plus a map of where/how it bites in that document so we can do a clean pass next.

---

## 1. Global style and formatting rules

**S1. Hyphens, dashes, and punctuation**

* Use only plain ASCII hyphens and commas. No em dashes or en dashes.
* When you currently have an em dash, decide case by case:

  * Replace with colon if you are introducing an explanation or list.
  * Replace with comma if it is just interrupting the sentence.
  * Example in Assignment 3 to fix:

    * "The mayor has minimal executive authority—one vote equal to other council members..."
      → "The mayor has minimal executive authority, with one vote equal to other council members..."
    * "Austin’s unique position as a technology hub—home to Dell, Oracle, Tesla..."
      → "Austin’s unique position as a technology hub, home to Dell, Oracle, Tesla..."

**S2. Section headings and numbering**

* Use simple numeric headings for major sections (1, 2, 3, etc.) and consistent capitalization.
* Avoid awkward line breaks like "Transfor- mation" in headings when you retype; keep "Transformation" on one line.

**S3. Timeline formatting**

When you later revise the Timeline section:

* Use a two-level structure:

  * Outer level: "I) 2025 October-November", "II) 2025 December-2026 January", etc.
  * Inner tasks: "10/01-11/15: Complete comprehensive literature review".
* Do not use date ranges like "October–November 2025" with special dashes; use "2025 October-November" at the phase level and "MM/DD-MM/DD" inside.

Applied to Assignment 3:

* Section 6 Timeline should be reformatted into:

  * I) 2025 October-November
  * II) 2025 December-2026 January
  * III) 2026 January-February
  * IV) 2026 February-March
  * V) 2026 March-May
    With each task inside using the MM/DD-MM/DD format.

**S4. Counts and notation**

* Do not use "n=15" style notation in narrative text.
* Instead, spell out: "15 semi-structured interviews (2 city officials or planners, 2 neighborhood association leaders, ...)".
* Applied to Assignment 3: In the Research Design Grid for Major Research Question 2, replace "Semi-structured interviews (n=15): – 3 city officials/planners ..." with plain English counts.

**S5. No informal notation or plus-sign blocks**

* Remove the "+ + + ..." scaffolding blocks and checklist-like artifacts from final text. Those are for our working notes only.

---

## 2. Data sources and temporal coverage rules

**D1. Always specify date ranges for each data source**

For each distinct dataset, explicitly state its temporal coverage in parentheses, in plain language:

* Opposition records (2007-2025)
* Development records (1966-2025; analysis uses 2007-2025)
* Travis Central Appraisal District parcel records (2021-2025; earlier years may be obtained via public information requests)
* Public testimony (2015-2025)
* Campaign finance data (2016-2025)
* Demographic context (2000-2025, at the tract level with decennial census and ACS updates)

Applied to Assignment 3:

* In the RQ1 grid, rewrite bullets as:

  * "Opposition records (2007-2025): protest petitions, valid petitions under Texas Local Government Code 211.006(d), and formal opposition filings extracted from zoning case files."
  * "City of Austin development records (1966-2025; analysis uses 2007-2025): site plan and zoning case data from the open data portal."
  * "Travis Central Appraisal District parcel-level appraisal and protest records (2021-2025; earlier years via public information requests)."
  * "Public testimony (2015-2025): City Council meeting transcripts and ATXN video recordings."
  * "Campaign finance data (2016-2025): City of Austin campaign finance datasets from the open data portal and related reports."
  * "Demographic context (2000-2025): census and ACS tract-level data."

**D2. Order of data sources**

When listing data sources, use a consistent priority order:

1. Opposition records
2. City of Austin development records
3. Travis Central Appraisal District records
4. Demographic context
5. Public testimony
6. Campaign finance data
7. Optional extras

Applied to Assignment 3:

* In the RQ1 Research Design Grid, reorder the list in that sequence instead of starting with TCAD.

**D3. Distinguish open data vs PIR**

* Explicitly flag which data are from the open data portal and which are obtained via public information requests (PIR).
* Examples:

  * "Travis Central Appraisal District records, obtained through public information requests."
  * "Campaign finance and development datasets from the City of Austin open data portal."
* Applied to Assignment 3:

  * Ensure TCAD records are clearly marked as PIR.
  * Ensure opposition records are clearly marked as coming from zoning case files, not the open data portal.
  * For open data, say "from the City of Austin Open Data Portal" once, not with long technical description.

**D4. Keep methodology details out of data-source bullets**

* In high-level data-source lists, do not embed:

  * Measurement rationales ("as a measure of civic engagement").
  * Technical details of extraction ("requires direct extraction from case files") unless absolutely necessary.
* Save those for the methods narrative.
* Applied to Assignment 3:

  * In the RQ1 grid, remove phrases like "Methodology: Manual reading of selected transcripts, skimming of large sample, automated analysis of remainder" from the Data Sources column and move them into the Details/Analysis column, or into a separate methods paragraph later.

**D5. No inline portal resource IDs**

* Do not write "resource ID g4yx-aw9r" in the body text.
* That detail belongs inside the reference or footnote, not in the prose of the assignment.

---

## 3. Temporal modeling and leakage rules

**M1. Define an expanding-window evaluation scheme**

Instead of a single "train 2018-2022, test 2023-2025" split, use an annual expanding-window forecast design, spelled out in long plain English.

Baseline scheme:

* Year 1: Train on all data up to and including 2018, test on 2019, validate on 2020-2025.
* Year 2: Train on all data up to and including 2019, test on 2020, validate on 2021-2025.
* Year 3: Train on all data up to and including 2020, test on 2021, validate on 2022-2025.
* Continue through the available data period.
* Final years:

  * Train on data up to and including 2023, test on 2024, validate on 2025.
  * Train on data up to and including 2024, test on 2025, predict 2026 (no validation possible).
  * Train on data up to and including 2025, predict 2026 and 2027 (no testing or validation possible).

Applied to Assignment 3:

* In the RQ1 Details/Analysis cell, replace "Validation on 2018-2022 cases, testing on 2023-2025" with a concise description of the expanding-window scheme above.

**M2. Handling pre-coverage years**

* For opposition cases before the first year of consistent TCAD coverage:

  * Hold parcel and tax variables fixed at their 2018 values for modeling, and clearly state this.
* Explicitly state that predictions and backtesting are only evaluated forward in time from the point at which all data sources converge (2018).
* Applied to Assignment 3:

  * Add a sentence in the RQ1 Details/Analysis:
    "For opposition cases prior to 2018, parcel and tax variables will be held fixed at their 2018 values to avoid temporal leakage, and predictive performance will only be evaluated in forward time from 2018 onward, when all core data sources align."

**M3. Rare-event classification metrics**

* Do not rely solely on accuracy, precision, recall, and ROC AUC, given that protest petitions are rare events.
* Add metrics appropriate for rare events, such as:

  * Precision-recall curves and area under the precision-recall curve.
  * F1 score for the positive (opposition) class.
  * Calibration curves and Brier score.
* Applied to Assignment 3:

  * In the RQ1 Details/Analysis, when you mention "Machine learning model to identify predictive factors," add:
    "Model performance will be evaluated using accuracy, precision, recall, and ROC AUC, as well as metrics more appropriate for rare events, including precision-recall curves, F1 scores for opposition cases, and calibration diagnostics."

---

## 4. Model class, functional form, and variable-importance rules

**F1. Explain nonparametric models in planning language**

* Replace phrases like "without strong functional form assumptions" with:

  * "without assuming a specific shape for the relationship between predictors and opposition."
* Provide a concrete, plausible curve example:

  * "For example, proximity effects may be highly nonlinear, with opposition spiking for developments within 500 feet and declining approximately exponentially beyond that distance."

Applied to Assignment 3:

* In any paragraph summarizing modeling (currently this is only briefly in the RQ1 grid), expand the description of nonparametric models with that kind of explanation and example.

**F2. Spell out model families**

* Use "tree-based methods such as Random Forests and Gradient Boosted Trees" instead of shorthand like "Tree-based methods (Random Forests, Gradient Boosted Trees/LightGBM)".
* Applied to Assignment 3:

  * When you first describe the model, add one clean sentence along those lines instead of a parenthetical list.

**F3. Variable importance and interpretation**

* Avoid "such as Variable importance metrics (permutation importance for tree-based models, coefficient magnitudes for logistic regression)" in parentheses.
* Instead, write:

  * "We will interpret model results using variable importance measures, including permutation importance for tree-based models and the magnitude and sign of coefficients in logistic regression."
* Applied to Assignment 3:

  * Add this in the RQ1 Details/Analysis cell when you describe analysis of predictors.

---

## 5. Qualitative design, sampling, and interviews

**Q1. Purposive sampling, not saturation framing**

* Describe sampling as purposive: you deliberately select participants to cover key stakeholder categories.
* Do not claim "thematic saturation" in advance for such a small sample.
* Do not include generic claims like "10-15 interviews typically suffice to identify major themes" unless you have a specific citation and really need it. In this project, you prefer to cut those lines entirely.

Applied to Assignment 3:

* In the RQ2 grid, phrase it as:

  * "Semi-structured interviews with approximately 15 participants, purposively sampled to include: 2 city officials or planners, 2 neighborhood association leaders, 3 housing advocates (for example, AURA members), 3 developers, and 3 civic technologists."

**Q2. No snowball sampling as primary method**

* If you mention snowball sampling, limit it to:

  * "Snowball referrals may be used to identify additional participants within these categories, up to the target numbers, but are not a primary recruitment strategy."
* In Assignment 3:

  * Currently you do not mention snowball sampling, so you do not need to add it. If you add it later, use the constrained wording above.

**Q3. Time commitment**

* Use a realistic time commitment of 15-30 minutes for interviews in this context.
* Only mention the time once, in the more detailed methodology section, not both in overview and again in detail.
* Assignment 3 currently does not state this; when you add a methodology paragraph, use "15-30 minutes" once.

**Q4. Align with IRB-approved interview guide**

* The core interview topics in the assignment must match the IRB-approved guide exactly.
* If you list core topics (as in your earlier artifact), ensure that:

  * They match the approved guide word-for-word, except where you are correcting contradictions already noted in this conversation.
* You requested: "confirm these align with the interview guide"; that check happens when we revise the full proposal text, not in this assignment. As a rule: do not introduce new topic categories here that are not in the IRB materials.

**Q5. Quotes and coding**

* Use "representative quotations" rather than "illustrative quotations".
* Clarify that the goal of selected quotes is to:

  * Ground interpretations in participants' own words.
  * Show typical patterns and, when relevant, important edge cases.
* Applied to Assignment 3:

  * In the RQ2 Details/Analysis cell, when you mention qualitative analysis, phrase it as:

    * "We will present representative quotations to ground key themes in participants' own words."

---

## 6. IRB, identifiers, and data management

**E1. Separation of identifiers and analytic dataset**

* Use the following structure (with or without the word "published", depending on IRB confirmation):

  * "Direct identifiers (names, addresses) will be stored separately from the analytic dataset, which will use coded study IDs to minimize privacy risks."
* If you want to include "published", place it as:

  * "from the analytic dataset that is used for analysis and publication".
* Do not split the phrase with an odd underscore or hyphen.

Applied to Assignment 3:

* In the methods or data section when you add IRB language, use exactly that wording and do not call it "PII" in the text; just "direct identifiers".

**E2. IRB implications vs standard pipeline**

* When you discuss data linkage and storage, distinguish between:

  * Standard engineering choices (keys, joins, pipelines).
  * IRB-relevant choices (where identifiers live, who has access, retention times).
* You do not need that full explanation inside Assignment 3, but the rule is: do not over-explain IRB logic when reviewers only need to know that identifiers are separate, access is limited, and coded IDs are used.

**E3. Investigator vs PI**

* Always label yourself as "Investigator", not "Principal Investigator".
* In Assignment 3, this only matters if your role appears explicitly; currently it does not, so we just keep the rule for later.

---

## 7. Site selection, policy context, and "so what"

**C1. Site selection logic order**

When you justify Austin as the site, present reasons in this order:

1. Uniqueness of opposition data:
   "Among the top 50 U.S. metros operating under similar state protest petition laws, Austin appears to be the only city that has systematically preserved protest petition and opposition records at the parcel or case level."
2. Natural experiment in reform vs stasis:
   "The study period includes both years of zoning stasis and years of significant reform (for example, parking elimination and the HOME Initiative), which together form a natural experiment in how zoning liberalization affects opposition patterns."
3. Technical and civic context:
   "Austin's role as a technology hub, with a large tech workforce and history of city-led data and tech initiatives, provides both the technical capacity and civic context to evaluate algorithmic tools."

Applied to Assignment 3:

* In the Introduction, you currently stress growth, reforms, and tech hub status. You should add the uniqueness-of-records point and re-order the reasons to match the sequence above.

**C2. HB 24 repeal context, but not in research design**

* Texas HB 24 (2025) substantially alters the protest petition law that generated your dataset (raising thresholds, removing automatic supermajority, etc.).
* This is important "so what" context, but you decided it does not belong in the research design sections of this assignment.
* Rule: reserve HB 24 for later "so what" or conclusion sections of the full thesis, not for Assignment 3's research design grid.

**C3. Reform timing language**

* If there is no widely agreed single pivot date from stasis to reform, do not pretend there is one.
* Instead of "concentrated November-December 2023 policy reforms", use:

  * "a reform period in late 2023, including the elimination of parking requirements and adoption of the HOME Initiative."
* Applied to Assignment 3:

  * In the Introduction, where you describe "sweeping reforms: eliminating parking requirements citywide (November 2023)...", this is fine; just avoid calling it a "concentrated" pivot point.

**C4. Tech culture citations**

* In longer versions of the proposal, you want at least one citation about Austin's tech landscape and receptivity to tech initiatives.
* Assignment 3 already references Axios Austin's piece "Our tech boom town". That is consistent with this rule; no extra change needed here.

---

## 8. Integration of quant and qual, fairness, and novelty

**I1. Neutral language about fairness and concerns**

* Do not pre-characterize stakeholder views as "overblown", "legitimate", or similar.
* Instead, use neutral formulations like:

  * "Qualitative data will help interpret whether stakeholders view these patterns as raising fairness concerns or as reflecting legitimate differences in political participation."

**I2. Accuracy and concern**

* The question "does higher accuracy make tools more useful or more concerning" is not novel in the general literature, but your novelty lies in applying it to predictive NIMBY opposition.
* Rule: frame your contribution as:

  * "first applying predictive modeling to parcel-level zoning opposition and pairing it with stakeholder interviews about democratic acceptability", rather than claiming conceptual novelty about "accuracy vs concern".
* Applied to Assignment 3:

  * In the "so what" section, if you mention accuracy and concerns, keep it short and tie it directly to this specific context rather than treating it as a new general theory.

**I3. Clarify what is being evaluated**

* Whenever you talk about democratic implications, be explicit about whether you are evaluating:

  * Use of the patterns descriptively (understanding opposition structure).
  * Using the patterns predictively (anticipating opposition).
* Applied to Assignment 3:

  * In Section 3 "Describe the 'so what'", you should clarify that your study evaluates both:

    * What the patterns themselves reveal about who participates in opposition.
    * The implications of using those patterns to anticipate opposition.

---

## 9. Citations, open data references, and external codebooks

**R1. Campaign finance datasets**

* Refer to campaign finance datasets succinctly:

  * "City of Austin campaign finance datasets from the Campaign Finance Data Portal (2016 onward)."
* Long descriptive blocks about views, column counts, and resource IDs belong in references or an appendix, not in the main text.

**R2. Codebook and external benchmarks**

* When you describe your qualitative coding strategy, note:

  * "There is no single, widely adopted off-the-shelf qualitative codebook for citizen perceptions of algorithmic governance in planning contexts; we draw on existing frameworks as conceptual starting points rather than importing a complete external codebook."
* Also note that this means you cannot benchmark your codebook against an external standard; you can only compare conceptually.
* Applied to Assignment 3:

  * In the RQ2 Details/Analysis cell (or a later methods section), if you mention deductive codes, explicitly include that "no single widely adopted codebook" line.

**R3. Deductive and inductive coding symmetry**

* Keep the deductive and inductive coding descriptions similar in length.
* Example structure:

  * Deductive: mention Grimmelikhuijsen and Meijer (input, throughput, output legitimacy), procedural justice themes, fairness, transparency, accountability, chilling effects.
  * Inductive: mention that additional themes will be generated from close reading of transcripts.
* Applied to Assignment 3:

  * When you expand qualitative analysis, ensure you do not write a long deductive paragraph and a one-line inductive paragraph; balance them.

---

## 10. Always-on meta-note and contradiction log

**N1. Travel to Austin**

* You asked to have this repeated: you must schedule time to be in Austin in person and get that travel covered by Columbia. That remains true and relevant for any in-person interviews or observations.

**N2. Contradiction logging**

* For any future full-proposal revision:

  * Keep track of any detail in this rule set that contradicts what is already in the earlier proposal artifact (for example, different training splits, sample sizes, or IRB phrasing).
  * When we revise the full proposal, we will systematically reconcile those contradictions.

For Assignment 3 specifically, the main contradictions to correct later are:

* The simple "2018-2022 train / 2023-2025 test" statement vs your preferred annual expanding windows.
* Any references to 45-60 minute interviews (if they appear in other documents) vs 15-30 minutes here.
* Any places where you are implicitly labeled PI rather than Investigator.

---

If you want, next step I can take this rule set and do a line-by-line edit of Assignment 3 into a clean, revised artifact that already obeys all of the above (including data ranges, reordered data sources, updated modeling description, and restructured timeline).

---

Meta-calibration footer
Persona: Senior planning and real-estate ML researcher.
Difficulty: Mostly in your zone of proximal development (rules you already implicitly use), with some stretch elements around making the evaluation design more publishable (expanding windows, rare-event metrics, codebook benchmarking) and slightly aspirational in the way the rules anticipate future thesis-level revisions.
One suggestion for your own framing: when you give long lists of micro-edits like this, grouping them into a small number of rule categories up front (style, data, modeling, qual, IRB, context) will make it easier for both you and future reviewers to see that you are enforcing a coherent standard rather than a set of one-off tweaks.



Title is Predicting NIMBYism, as in the IRB


















confirm everything has been written to the specified page lengths. del pdf and recompile pdflatex twice


has the outline with page lengths been followed? where is the IRB appendix materials at the end?


the appendix should include the verbatim interview gudie directly lifted if pretty printed from the IRB.pdf

"\textit{Note: The following guide is based on the topics approved in IRB Protocol ACYY0820.}"








if this should now be integrated and is no longer separately necessary given the outline we are now following do so

2.1 Significance and ”So What”
This research contributes to literature on algorithmic governance and housing policy by separat-
ing the empirical structure of opposition from the normative question of prediction. It provides
evidence on the tradeoffs between efficiency and legitimacy, informing theoretical debates about
technocratic city management.

double check all page lengths for different sections against the outline i provided in earlier prompt and report back to me

why did you abbreviate "Continues annually" I wanted every single one written out
















"or similar
pro-housing organizations" 1 member of pro-housing organization such as AURA 1 of pro-community organization such as you search em up
"or similar
pro-housing organizations"






















Use consisten APA citation throughout don't switch between LaTeX and APA
"idc2024"















confirm to me these are verbatim from the IRB.pdf and everything Dory requested (see my first prompt)
"Interview Guide"













You are still at 18 pages not including the appendices. Follow the Outline pages per 

UP Thesis Proposal Outline | Dory Thrasher’s Group 
The key deliverable for this semester is a complete proposal for your UP thesis. The proposal does three 
main things:  
1. It describes your topic and lays out your research questions, explaining why the research has 
value to planners. (What do you want to know, and why?) 
2. It reviews the relevant literature in order to situate your research project within the field. 
(Where does your work fit in as it relates to topic, methodology, and theoretical approach?) 
3. It serves as a fairly comprehensive outline for conducting the research. (What are you going to 
do to answer your research questions?) 
The proposal should be about 10 pages double-spaced, and will be organized as follows: 
Introduction (1-2 pages). The introduction describes the purpose of the thesis. It describes the topics, 
presents the research question, and explains why the research is of interest to planners 
Background (1-2 pages). This section describes the history of the event, public problem, or program 
that you are studying. For example, if you are investigating who uses a public bike share system, you 
would discuss the history of cycling in the city, the development of the public bike share system 
including funding and network planning, the known issues, concerns, or debates about the system. This 
section should narratively build towards your research questions. 
Literature review (2-3 pages). Here you examine previous (predominately academic) research on your 
topic. Focus on the questions asked, the research findings, and the methodologies used. It is important 
to be aware of the point of view of the literature (e.g. bike network planning, vs transportation equity, 
vs. economic development, vs. politics).  
Research design (2-3 pages). In this section you describe your research approach (e.g. quantitative data 
analysis, interviews, surveys, case study), discuss your selection of a research site and research 
subjects, your approach to sampling or recruiting, where data will be obtained, and your plan for 
analysis 
Task schedule (1 page). This is your schedule for key research and writing tasks allowing you to meet 
the deadlines for on-time submission of your complete thesis. 
References (2-3 pages). List all your sources. Use a consistent citation style.  


not just "Interview Guide" but all appendices follow the IRB.pdf materials

pdflatex more frequently so I can see progress

fix this citation "austin ̇campaign ̇finance ̇transactions; austin ̇campaign ̇finance"
















did you fold in "Significance and ”So What”
This research contributes to literature on algorithmic governance and housing policy by separat-
ing the empirical structure of opposition from the normative question of prediction. It provides
evidence on the tradeoffs between efficiency and legitimacy, informing theoretical debates about
technocratic city management."

















it seems you lost what we had for the following i have pasted it below so you can reinsert and trim it

"Predicting NIMBYism
Daniel Lewis
dl3645@columbia.edu
Columbia University
Urban Planning
December 10, 2025
1 Introduction
This thesis investigates the patterns of neighborhood opposition to housing development in Austin,
Texas, and evaluates the democratic implications of using predictive analytics to anticipate such
behavior. The project asks: how can cities identify and understand patterns of neighborhood op-
position, and what are the normative consequences of using algorithmic tools to manage this par-
ticipation?
Austin, Texas faces an unprecedented housing affordability crisis that has fundamentally re-
shaped its political landscape. Between 2010 and 2020, Austin’s population grew by 33%, adding
nearly 250,000 residents (U.S. Census Bureau, 2020). This growth, driven primarily by the tech-
nology sector, created severe supply constraints. By May 2022, the median home price reached
$550,000, up 79% from January 2020. As of August 2025, prices have moderated to $444,490 but
remain unaffordable for most residents (UnlockMLS, 2025).
The research addresses a critical gap at the intersection of urban planning and algorith-
mic governance. As cities turn to technological solutions to streamline development, the use of
1
predictive tools raises questions about democratic participation and civil liberties. The findings
will directly inform policy in Austin—where Development Services could use insights to improve
engagement—and establish ethical frameworks for other cities.
Research Questions
The project is guided by three questions:
a. What demographic, geographic, and behavioral factors predict whether residents will oppose
specific development projects?
b. How do different stakeholders perceived the legitimacy and utility of algorithmic tools in
housing policy?
c. What alternative technological approaches could improve development processes without
raising concerns about democratic participation?
An important recent development shapes this work. In 2025, the Texas Legislature passed
House Bill 24, raising the valid petition protest threshold from 20% to 60% and eliminating the
automatic supermajority requirement (effective Sept 1, 2025). While these changes occurred after
our study period (2018-2025), they create crucial context: we are analyzing behavior under a
regime that has now been substantially weakened but continues to operate in other states.
2 Background
Austin’s governance structure complicates housing policy implementation. The city operates under
a council-manager system where the mayor has limited executive authority. Passing controversial
reforms has historically required council supermajorities to overcome property owner protests au-
thorized by Texas law (City of Austin Planning Department, 2010).
Historically, neighborhood associations have mounted organized opposition to develop-
ment using tools like valid petitions and public testimony. This opposition often correlated with
2
neighborhood demographics, raising questions about equity. In response to the deepening cri-
sis, Austin underwent a political transformation. The November 2022 elections delivered a pro-
housing supermajority (Travis County Clerk, 2022). This shift enabled sweeping reforms: elimi-
nating parking requirements citywide in November 2023 (Axios Austin, 2023), reducing minimum
lot sizes, and allowing density on corner lots through the HOME Initiative (Pool, 2023).
These reforms have generated measurable impacts, with apartment vacancy rates climbing
to 10-15% by 2025 (KVUE, 2024). However, implementation challenges persist, with permit
volumes fluctuating and staffing pressures in the Development Services Department. Throughout
this period, opposition tactics have evolved. Austin’s unique position as a technology hub—home
to major tech firms and a workforce of 180,000—provides a distinct civic context for debating the
role of data in managing this conflict.
2.1 Significance and ”So What”
This research contributes to literature on algorithmic governance and housing policy by separat-
ing the empirical structure of opposition from the normative question of prediction. It provides
evidence on the tradeoffs between efficiency and legitimacy, informing theoretical debates about
technocratic city management."











===========================================
CONVERSATION SESSION: December 11, 2025
===========================================

did you actually search externally the factuality of each?

fix it

now delete critical corrections

write up the approach we just took then apply it to cehck Completion Summary against both Background and Submitted Thesis

push Thesis Proposal.txt into Deprecated

confirm that you actually search externally thos now correct entries

and for both the previous critical corrections and these are we citing those same references you just pulled? did you check the references we are now using? decide which between the two if they contain the same information is the better reference to keep citing in-line (though keeping both in the master references.bib)?

they may be under different keywords than you used to search

add the missing bib entries and ensure where they are cited inline these citations are used, notify me if doing so would potentially deprecate an existing citation for the same fact

defer the full audit for later by putting in a master TODO file at the top level

rename prompts.txt to Prompts.txt
irb content to IRB-SUBMITTED.txt

does the background LONG contain all the tables and figures?
now delete completion summary

is every Austin_Housing_Summary included in references.bib?
no instead place in the TODOs a check tables and figures when drafting thesis in the spring, that these tables and figures are not included in the background long
create a new folder named Thesis_Draft and rename Submitted_Thesis_Proposal to Thesis_Proposal-SUBMITTED

check that every sources mentioned in Austin Housing Bibliography is already included in the references.bib

fix this in references
lines 2847-3004

use greps to adopt a consistent formatting scheme throughout references.bib
i prefer this one
@article{zarsky2016,
  author       = {Zarsky, Tal Z.},
  year         = {2016},
  title        = {The trouble with algorithmic decisions: An analytic road map to examine efficiency and fairness in automated and opaque decision making},
  journaltitle = {Science, Technology, \& Human Values},
  volume       = {41},
  number       = {1},
  pages        = {118--132},
  doi          = {10.1177/0162243915605575}
}

TODO: organize references.bib to follow a similar commented organizational scheme as Austin Housing Bibliography

move into thesis daft  Backgroun LONG IRB SUBMITTED Master Sources Prompts references.bib Tables and Figures and  a copy of Thesis Proposal Submitted tl dir itself
Rename Tables and Figures Table_and_Figures-TO_INTEGRATE

mkdir Assignments_and_Proposal-SUBMITTED and move Assignments into it renaming it Assignments-SUBMITTED and Thesis-Proposal

Move Austin_Housing_Bibliography into Thesis_Draft
put in TODO rename that file and decide what to do with it

place a raw copy of all my prompts through this entire conversation appended to Prompts

## Session: December 11, 2025 (19:04-19:18 CST)

### Prompts

1. ensure you are only appending to the changelog not revising in full, mark what we have in there thus far as reviewed and anything newly appended as UNREVIEWED and insert these rules into the guidelines

2. as well as perennial TODO for me to make such reviews, in the appropriate place for prioritization, renumbering/orging the rest as necessary

3. is there an obvious relationship between these deprecated writings? can they be marked with a number to indicate the order in which they were produced? if not let me know if so number them prepended to each one 1-, 2-, etc just the directories

4. move all materials in Thesis_Draft into own subdir named Thesis_Draft_Reference_Materials

5. did you come up with that numbering by examining the contents of the .tex files of each or just from the titles?

6. is that evidenced from any discernible revisions to the contents themselves, rather than the titles?

7. search deeper to see if there is an assignable order otherwise label the parallel ones with '*.1, *.2' formatting scheme, with 1 weighted towards whichever appears first

8. explain numbering scheme in brief readme specific to that folder

9. append new prompts to prompts comprehensive and to own prompts file named medium quality

10. are there any other TODOs we haven't yet placed into the TODO from what you have seen thus far? move all Prompts into own Prompts.d ensure recent prompts are reflected in GUIDELINES

11. readme every directory and subdirectory

12. top level readme Daniel Hardesty Lewis' Master's thesis for Urban Planning at Columbia University on Predicting NIMBYism using machine learning models, using Austin rezoning protest letters data from 2007-2025, validated from 2018-2025

13. title is simply Predicting NIMBYism

14. final set of prompts, changes, guidelines, etc if haven't already

## 2025-12-11T21:49:00 - Guidelines and TODO Refinement

1. "include in TODOs download IRB.pdf, UP Outline and UP Thesis Guidelines documents from Rascal & Dory's email"
2. "fix your guidelines so you are always inserting any new TODOs into appropriatemost place prioritized, not up top or at bottom"
3. (Comment) "and do so"
4. "and correct the TODO.md"
5. (Comment) "likewise dont simply put this or any other insertions at the end put in the best prioritized place and renumber approriate ensure that rule is reflected in the guidelines"
6. "confirm TODO management is prioritized 7 and no higher"
7. (Comment) "review all prompts made in the conversation and that there are no remaining incomplete prompts"
8. "confirm that to me by writing out all prompts to this conversation"
9. "ensure that is adequately reflected in its relevant guideline"
10. "any GUIDELINES worth importing?

# HW3 Project Guidelines

> **CRITICAL**: When appending to `CHANGELOG.md`, always include the current timestamp (date and time).

## P1 - Critical (Every Session)

### 1. Assignment Compliance
- **Verify against `hw3.txt`**: Ensure every section explicitly answers prompt questions (a)-(d).
- **Check constraints**: Word counts (if any), page limits, required sections.
- **Fact Verification**: Ensure claims about model performance (RMSE, R^2) match `hw3_verified.txt` or notebook outputs.

### 2. Citation Integrity
- **Verify inline citations**: Ensure `\cite{}` keys match `bib` entries.
- **Source Priority**:
  1. Course materials / Textbook (Murphy)
  2. Original methodology papers (MIWAE, VAE)
  3. Domain papers (NYC Real Estate, Berry & Bednarz)

---

## P2 - Important (Regular Maintenance)

### 3. Bibliography Management
#### 3.1 Formatting Standard
```bibtex
@article{key,
  author = {Name},
  year = {2025},
  title = {Title}
}
```
- Blank line between each field.
- Verify consistent use of BibTeX keys.

### 4. Documentation
#### 4.1 Session Tracking
- Record major decisions in `CHANGELOG.md`.
- **Append ONLY**: Add new entries at the end.
- **Review Status**: Mark entries as `(REVIEWED)` or `(UNREVIEWED)`.

---

## P3 - Housekeeping

### 6. Task Management
- **Prioritized Insertion**: When adding new TODO items, insert them into the appropriate priority section (P1/P2/P3) based on urgency and importance. Do NOT append them to the top or bottom of the file."
11. "make copy of Prompts into tldir PROMPTS-LOG.md and insert these most recent PROMPTS ensure every new insertion henceforth is timestamped and there is a GUIDELINE for this"
12. "still havent organized 

### Additional High Priority TODOs


- [ ] (UNREVIEWED) Add a README to every sub‑directory that still lacks one (e.g., `Prompts.d/`, `Background‑COMPREHENSIVE.d/`, any newly created LaTeX project folders)
- [ ] (UNREVIEWED) Create a short README inside `Prompts.d/` describing the quality‑tier system (Highest_Quality, Medium_Quality, etc.)
- [ ] (UNREVIEWED) Add a checklist item to review the new “timestamp‑only when appending to CHANGELOG” rule after the next commit
- [ ] (UNREVIEWED) Verify that the root `README.md` contains a link to the project‑wide `GUIDELINES.md`
- [ ] (UNREVIEWED) Run a quick sanity‑check that every `.tex` file now contains the author name “Daniel Hardesty Lewis”

still havent organized 

### Additional High Priority TODOs
- [ ] (UNREVIEWED) Add Title to LaTeX document: "Semi-Supervised Latent Variable Model for NYC Property Valuation"
- [ ] (UNREVIEWED) Verify all P1 critiques in `critiques.md` are marked complete
- [ ] (UNREVIEWED) Compile and verify final PDF `hw3-dl3645.pdf`

directly into the prioritized ones below, isn't there a guideline for this?"
13. "follow the same organization for completed TODO as for TODO and and they should be timestamped and that should be GUIDELINEd for now provide current timestamp for all of them, going forward you know what to do"
14. "perform your wrap up tasks, define for me what those are, and ensure they are reflected as such in the guidelines"
15. "provide a more consistent structure to any logs inserted from here and never ellispis any logs (...) ensure reflected in the GUIDELINEs then wrap up again, wrap up should include not just Git Status: Check status. but final commit"
16. "anything else to sync from these other GUIDELINES? ..."
17. "perform wrapup"
18. "provide insertion time for ... if unknown just now, ensure that is in a GUIDELINE"
19. "wrap up"

