# Thesis Empirical Pipeline

This directory contains the core sequence of data harvesting, NLP transcription, and panel assembly scripts required to reconstruct the raw analytical `Warehouse`. 

*Note: The top-level orchestrator (`run_thesis.py`) intentionally bypasses this directory to ensure rapid compute reproducibility since rebuilding the spatial and NLP panels from scratch requires heavy computational hardware and API keys. To run end-to-end, users must execute these directories sequentially.*

### Directory Structure

1. **`01_Scrapers_and_Harvesters`**
   - Retrieves unstructured files (PDFs, MP4s, HTML agendas) and raw json.
   - Core dependency: City of Austin Public Portal & Travis County Tax Assessor.

2. **`02_Transcription_and_NLP`**
   - Handles Whisper AI audio-to-text models and extracts speaker diarization.
   - Featurizes textual meeting transcripts into dense TF-IDF and NLP embeddings for Stage $H_3$ analysis.

3. **`03_Data_Engineering_and_Panel_Builds`**
   - The master joining layer. Executes spatial bounding integrations (H3 index hashing) to merge Tax Assessor parcels with City Planning polygons. 
   - Generates the rigorous temporal snapshots ($H_0$, $H_1$, $H_3$) ensuring zero look-ahead bias prior to modeling.

4. **`04_Cloud_Orchestration`**
   - Remote computing pipelines (e.g., executing the V-REx operations or heavy Whisper models on Modal or AWS GPUs).
