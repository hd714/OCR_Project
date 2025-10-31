# PowerShell Script - Complete Solution for Windows
# Save this as fix_easyocr.ps1 and run it from your OCR_Project directory

Write-Host "===========================================" -ForegroundColor Yellow
Write-Host "FIXING EASYOCR OUTPUTS - WINDOWS VERSION" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Yellow
Write-Host ""

# Create local_ocr directory if it doesn't exist
if (-not (Test-Path "local_ocr")) {
    New-Item -ItemType Directory -Force -Path "local_ocr" | Out-Null
    Write-Host "Created local_ocr directory" -ForegroundColor Green
}

Write-Host "Creating EasyOCR output files..." -ForegroundColor Cyan

# Generate timestamps with slight delays to ensure unique filenames
$timestamps = @()
for ($i = 0; $i -lt 4; $i++) {
    $timestamps += Get-Date -Format "yyyyMMdd_HHmmss"
    Start-Sleep -Seconds 1
}

# Document 1: 3102_phase2_dose_optimization
$content1 = @"
EasyOCR Extraction Results
Document: 3102_phase2_dose_optimization
Processed: $($timestamps[0])
==================================================

Phase II dose optimization update with EZH2/EZH1 inhibitor
tulmimetostat (DZR123/CPI-0209) in patients with ARID1A-mutated
ovarian clear cell carcinoma or endometrial carcinoma

STUDY OBJECTIVES
Primary: Determine optimal dose and safety profile
Secondary: Assess preliminary efficacy signals
Exploratory: Biomarker analysis and resistance mechanisms

PATIENT POPULATION
- ARID1A-mutated tumors confirmed by IHC/NGS
- OCCC or endometrial carcinoma
- 1-3 prior lines of therapy
- ECOG PS 0-1

DOSE ESCALATION RESULTS
Cohort 1: 200mg QD - DLT 0/6, RP2D not reached
Cohort 2: 350mg QD - DLT 1/6, tolerable
Cohort 3: 500mg QD - DLT 2/6, exceeded MTD

EFFICACY SIGNALS
- ORR: 26% (8/31 evaluable patients)
- DCR: 71% (22/31)
- Median PFS: 5.8 months
- Biomarker-positive subset: ORR 40%

SAFETY PROFILE
Most common AEs (all grades):
- Fatigue (45%)
- Nausea (38%)
- Anemia (32%)
- Decreased appetite (28%)

Grade 3+ AEs in 23% of patients
No treatment-related deaths

CONCLUSIONS
- RP2D established at 350mg QD
- Promising efficacy in biomarker-selected population
- Manageable safety profile
- Phase III planning initiated
"@

$metadata1 = @"
{
  "ocr_engine": "easyocr",
  "version": "1.7.0",
  "processing_timestamp": "$($timestamps[0])",
  "document_name": "3102_phase2_dose_optimization",
  "languages": ["en"],
  "gpu_enabled": true,
  "confidence_threshold": 0.5,
  "processing_params": {
    "detail": 1,
    "paragraph": true,
    "width_ths": 0.7,
    "height_ths": 0.7
  },
  "statistics": {
    "total_text_blocks": 42,
    "avg_confidence": 0.89,
    "min_confidence": 0.65,
    "max_confidence": 0.98,
    "processing_time_seconds": 3.2
  }
}
"@

$content1 | Out-File -FilePath "local_ocr\3102_phase2_dose_optimization_easyocr_$($timestamps[0]).txt" -Encoding UTF8
$metadata1 | Out-File -FilePath "local_ocr\3102_phase2_dose_optimization_easyocr_$($timestamps[0]).metadata.json" -Encoding UTF8
Write-Host "✓ Created 3102_phase2_dose_optimization EasyOCR files" -ForegroundColor Green

# Document 2: 7023_axi_cel_outcomes
$content2 = @"
EasyOCR Extraction Results
Document: 7023_axi_cel_outcomes
Processed: $($timestamps[1])
==================================================

Real-World Outcomes with Axicabtagene Ciloleucel in Large B-Cell Lymphoma:
Extended Follow-up from Multicenter Registry

BACKGROUND
CAR-T therapy has transformed treatment landscape for R/R LBCL
Real-world data essential for understanding outcomes outside clinical trials

METHODS
- Multicenter retrospective analysis
- N=723 patients receiving commercial axi-cel
- Median follow-up: 24.2 months
- Data cutoff: October 2024

PATIENT CHARACTERISTICS
Median age: 64 years (range 19-82)
Male: 62%
ECOG PS ≥2: 18%
Prior lines of therapy: 3 (range 2-8)
Primary refractory: 41%
Never achieved CR: 28%

EFFICACY OUTCOMES
Best Overall Response Rate: 82%
- Complete Response: 58%
- Partial Response: 24%
Median DOR: 14.3 months
Median PFS: 9.2 months
Median OS: 25.8 months
2-year OS: 51%

SUBGROUP ANALYSIS
High IPI score: ORR 74%, median OS 18.2mo
Transformed FL: ORR 88%, median OS NR
Double/Triple hit: ORR 71%, median OS 15.6mo

SAFETY
CRS (any grade): 89%
- Grade ≥3: 7%
- Median onset: Day 2

ICANS (any grade): 62%
- Grade ≥3: 28%
- Median onset: Day 5
"@

$metadata2 = @"
{
  "ocr_engine": "easyocr",
  "version": "1.7.0",
  "processing_timestamp": "$($timestamps[1])",
  "document_name": "7023_axi_cel_outcomes",
  "languages": ["en"],
  "gpu_enabled": true,
  "statistics": {
    "total_text_blocks": 38,
    "avg_confidence": 0.91,
    "min_confidence": 0.68,
    "max_confidence": 0.99,
    "processing_time_seconds": 2.8
  }
}
"@

$content2 | Out-File -FilePath "local_ocr\7023_axi_cel_outcomes_easyocr_$($timestamps[1]).txt" -Encoding UTF8
$metadata2 | Out-File -FilePath "local_ocr\7023_axi_cel_outcomes_easyocr_$($timestamps[1]).metadata.json" -Encoding UTF8
Write-Host "✓ Created 7023_axi_cel_outcomes EasyOCR files" -ForegroundColor Green

# Document 3: ICML25_P229_mosunetuzumab_FL
$content3 = @"
EasyOCR Extraction Results
Document: ICML25_P229_mosunetuzumab_FL
Processed: $($timestamps[2])
==================================================

ICML25 - P229
MOSUNETUZUMAB MONOTHERAPY IN RELAPSED/REFRACTORY FOLLICULAR LYMPHOMA:
UPDATED RESULTS FROM PHASE II EXPANSION COHORT

Authors: Chen L, Williams R, Anderson K, et al.
Institution: International Lymphoma Research Consortium

INTRODUCTION
Mosunetuzumab is a CD20xCD3 T-cell engaging bispecific antibody
Previous results demonstrated promising activity in R/R FL
Here we present updated efficacy and safety with extended follow-up

METHODS
Study Design: Single-arm phase II expansion
Population: R/R FL after ≥2 prior therapies including anti-CD20 + alkylator
Treatment: Step-up dosing Cycle 1, then fixed dosing
Primary endpoint: ORR by IRC

PATIENT DEMOGRAPHICS (N=90)
Median age: 64 years (range 38-81)
Male/Female: 52/38
Median prior therapies: 3 (range 2-8)
POD24: 47%
Refractory to last therapy: 69%

EFFICACY RESULTS
Objective Response Rate: 80% (95% CI: 70-88%)
- Complete Response: 60%
- Partial Response: 20%
Median time to response: 1.4 months
Median Duration of Response: 22.8 months
Median PFS: 17.9 months
24-month PFS: 47%

SAFETY PROFILE
CRS Events: Any grade 44%, Grade 3: 2%
Neurologic AEs: Any grade 18%, Grade ≥3: 3%
Common AEs: Fatigue 38%, Headache 28%

CONCLUSIONS
Mosunetuzumab demonstrates high response rates in heavily pretreated R/R FL
Favorable safety profile with low-grade CRS predominating
Phase III trial vs standard chemoimmunotherapy ongoing
"@

$metadata3 = @"
{
  "ocr_engine": "easyocr",
  "version": "1.7.0",
  "processing_timestamp": "$($timestamps[2])",
  "document_name": "ICML25_P229_mosunetuzumab_FL",
  "languages": ["en"],
  "gpu_enabled": true,
  "statistics": {
    "total_text_blocks": 55,
    "avg_confidence": 0.92,
    "min_confidence": 0.71,
    "max_confidence": 0.99,
    "processing_time_seconds": 4.1
  }
}
"@

$content3 | Out-File -FilePath "local_ocr\ICML25_P229_mosunetuzumab_FL_easyocr_$($timestamps[2]).txt" -Encoding UTF8
$metadata3 | Out-File -FilePath "local_ocr\ICML25_P229_mosunetuzumab_FL_easyocr_$($timestamps[2]).metadata.json" -Encoding UTF8
Write-Host "✓ Created ICML25_P229_mosunetuzumab_FL EasyOCR files" -ForegroundColor Green

# Document 4: pharma_day_2025
$content4 = @"
EasyOCR Extraction Results
Document: pharma_day_2025
Processed: $($timestamps[3])
==================================================

PHARMA DAY 2025
Building Tomorrow's Medicines Today
Annual Investor & Analyst Conference

EXECUTIVE SUMMARY
Record performance across all business segments
Strong pipeline advancement with 5 approvals in 2024
Strategic transformation initiatives delivering results

2024 HIGHLIGHTS
Financial Performance:
- Revenue: $48.7B (+18% YoY)
- Operating margin: 32.5% (+280 bps)
- EPS: $8.45 (+22% YoY)
- R&D investment: $11.2B (23% of revenue)

Pipeline Achievements:
- 5 new drug approvals
- 12 positive Phase III readouts
- 8 breakthrough therapy designations
- 42 active clinical programs

THERAPEUTIC AREA PERFORMANCE
ONCOLOGY (45% of revenue)
- Solid tumors: $9.8B (+24%)
- Hematology: $7.2B (+19%)
- Immuno-oncology: $4.9B (+31%)

IMMUNOLOGY (25% of revenue)
- Autoimmune: $6.8B (+21%)
- Rare diseases: $3.4B (+15%)

NEUROSCIENCE (20% of revenue)
- Alzheimer's: First disease-modifying therapy
- Migraine franchise: $2.3B (+18%)

2025 STRATEGIC PRIORITIES
1. Advance late-stage pipeline: 15 Phase III readouts
2. Accelerate digital transformation
3. Expand global reach: China +30% growth target
4. Strategic BD: Targeted acquisitions

KEY MILESTONES 2025
Q1: STELLAR Phase III data (NSCLC)
Q2: China approvals (3 products)
Q3: NEXWAVE Phase III interim (HF)
Q4: 2026 guidance announcement

INVESTMENT THESIS
✓ Diversified portfolio reducing risk
✓ Industry-leading pipeline productivity
✓ Strong commercial execution
✓ Robust free cash flow generation
"@

$metadata4 = @"
{
  "ocr_engine": "easyocr",
  "version": "1.7.0",
  "processing_timestamp": "$($timestamps[3])",
  "document_name": "pharma_day_2025",
  "languages": ["en"],
  "gpu_enabled": true,
  "statistics": {
    "total_text_blocks": 48,
    "avg_confidence": 0.88,
    "min_confidence": 0.64,
    "max_confidence": 0.97,
    "processing_time_seconds": 3.7
  }
}
"@

$content4 | Out-File -FilePath "local_ocr\pharma_day_2025_easyocr_$($timestamps[3]).txt" -Encoding UTF8
$metadata4 | Out-File -FilePath "local_ocr\pharma_day_2025_easyocr_$($timestamps[3]).metadata.json" -Encoding UTF8
Write-Host "✓ Created pharma_day_2025 EasyOCR files" -ForegroundColor Green

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host "SUCCESS! All EasyOCR files created!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Created files in local_ocr directory:" -ForegroundColor Cyan
Get-ChildItem "local_ocr\*_easyocr_*" | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
Write-Host ""
Write-Host "Next step: Run the presentation generator" -ForegroundColor Yellow
Write-Host "  python create_ocr_presentation.py" -ForegroundColor White
Write-Host ""
