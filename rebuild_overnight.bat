@echo off
REM rebuild_overnight.bat — Full dataset rebuild pipeline
REM Run from repo root: rebuild_overnight.bat
REM Uses .bat to avoid PowerShell stderr-as-error issues

echo [%date% %time%] Starting full dataset rebuild pipeline
echo [%date% %time%] Starting full dataset rebuild pipeline > logs\rebuild_overnight.log

REM ── STEP 1: Collect from ccdv (threshold 0.35, all splits) ──────
echo.
echo ============================================
echo STEP 1: Collect CS papers from ccdv/arxiv-summarization
echo ============================================
python src/processing/collect_cs_papers.py --target 2500 --threshold 0.35 >> logs\rebuild_overnight.log 2>&1
if errorlevel 1 (
    echo STEP 1 FAILED — check logs\rebuild_overnight.log
    goto :end
)
echo STEP 1 COMPLETE

REM ── STEP 2: Supplement from jamescalam/ai-arxiv ─────────────────
echo.
echo ============================================
echo STEP 2: Collect supplemental papers from jamescalam/ai-arxiv
echo ============================================
python src/processing/collect_ai_arxiv.py >> logs\rebuild_overnight.log 2>&1
if errorlevel 1 (
    echo STEP 2 FAILED — check logs\rebuild_overnight.log
    goto :end
)
echo STEP 2 COMPLETE

REM ── Count papers ────────────────────────────────────────────────
echo.
python -c "import os; n=len([f for f in os.listdir('data/raw/papers') if f.endswith('.json') and not f.startswith('_')]); print(f'Total papers collected: {n}')"

REM ── STEP 3: Re-run pair mining ──────────────────────────────────
echo.
echo ============================================
echo STEP 3: Cross-modal pair mining
echo ============================================
if exist data\processed\cross_modal_pairs\pairs.json del data\processed\cross_modal_pairs\pairs.json
python src/processing/pair_miner.py --no-category-filter >> logs\rebuild_overnight.log 2>&1
if errorlevel 1 (
    echo STEP 3 FAILED — check logs\rebuild_overnight.log
    goto :end
)
echo STEP 3 COMPLETE

REM ── STEP 4: Quality filter ─────────────────────────────────────
echo.
echo ============================================
echo STEP 4: Quality filter
echo ============================================
if exist data\processed\manifest.json del data\processed\manifest.json
python src/processing/quality_filter.py >> logs\rebuild_overnight.log 2>&1
if errorlevel 1 (
    echo STEP 4 FAILED — check logs\rebuild_overnight.log
    goto :end
)
echo STEP 4 COMPLETE

REM ── STEP 5: HDF5 dataset build ─────────────────────────────────
echo.
echo ============================================
echo STEP 5: HDF5 dataset build
echo ============================================
if exist data\hdf5\engineering.h5 del data\hdf5\engineering.h5
python src/processing/dataset_builder.py >> logs\rebuild_overnight.log 2>&1
if errorlevel 1 (
    echo STEP 5 FAILED — check logs\rebuild_overnight.log
    goto :end
)
echo STEP 5 COMPLETE

REM ── STEP 6: Validation ─────────────────────────────────────────
echo.
echo ============================================
echo STEP 6: Dataset validation
echo ============================================
python src/processing/validate_dataset.py --samples 10 >> logs\rebuild_overnight.log 2>&1
if errorlevel 1 (
    echo STEP 6 FAILED — check logs\rebuild_overnight.log
    goto :end
)
echo STEP 6 COMPLETE

echo.
echo ============================================
echo PIPELINE COMPLETE
echo ============================================
echo Check logs\rebuild_overnight.log for full output
echo Scroll to bottom for validation report

:end
echo.
echo [%date% %time%] Pipeline finished >> logs\rebuild_overnight.log
echo [%date% %time%] Pipeline finished
pause