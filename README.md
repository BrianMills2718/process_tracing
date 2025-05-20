# Process Tracing Toolkit

This toolkit is designed for qualitative analysis of causal processes in social science research. It provides tools to extract causal information from text, visualize it as a network, and perform theoretical analysis based on process tracing methodology.

## Project Structure

```
process_tracing/
├── core/                  # Core Python modules (analyze.py, extract.py, ontology.py, etc.)
├── input_text/            # Input text files, organized by project
│   └── revolutions/
│       └── american_revolution.txt
├── output_data/           # All outputs, organized by project and timestamp
│   └── revolutions/
│       ├── revolutions_YYYYMMDD_HHMMSS_graph.json
│       ├── revolutions_YYYYMMDD_HHMMSS_graph.html
│       └── revolutions_YYYYMMDD_HHMMSS_analysis.md
├── README.md
└── ...
```

## Core Components

The toolkit is organized into several key directories and scripts:

### 1. `core/` - Core Logic
Contains the fundamental scripts for data extraction, ontology, and analysis.

*   `core/extract.py`:
    *   **Purpose**: Extracts structured causal graph data from input text using Google's Generative AI (Gemini).
    *   **Input**: Text file(s).
    *   **Output**: JSON file representing the causal graph and an HTML visualization of this graph.
    *   **Key Features**: Uses a comprehensive ontology (defined in `core/ontology.py`) for node and edge types, robust JSON parsing and validation, dynamic configuration for input/output paths.

*   `core/analyze.py`:
    *   **Purpose**: Core analysis engine for process tracing networks.
    *   **Input**: JSON file containing the causal graph.
    *   **Output**: Analysis reports in Markdown or HTML format with visualizations.
    *   **Key Features**: Causal chain identification, mechanism evaluation, evidence analysis, condition analysis, actor analysis, alternative explanation analysis, and network metrics calculation.

*   `core/ontology.py`:
    *   **Purpose**: Defines the schema for the process tracing graphs.
    *   **Contents**: Specifies `NODE_TYPES` (e.g., Event, Causal_Mechanism, Actor) with their required/optional properties, and `EDGE_TYPES` (e.g., causes, enables, tests_hypothesis) with their source/target constraints. Also includes `NODE_COLORS` for visualization.

### 2. `input_text/` - Input Data
Directory containing the text files to be analyzed. These files should contain the narrative or case study text from which causal relationships will be extracted.

### 3. `output_data/` - Output Files
Directory containing all generated output files:

*   `json/`: Contains the extracted causal graph data in JSON format.
*   `reports/`: Contains the generated analysis reports in Markdown or HTML format.
*   `charts/`: Contains any generated visualization charts in PNG format.

### 4. `process_trace.py` - Main Entry Point
The main script that orchestrates the entire process:
1. Reads input text files
2. Extracts causal relationships
3. Generates the causal graph
4. Performs analysis
5. Produces reports and visualizations

## Usage

1. **Add a new project:**
   - Place your input `.txt` files in a subdirectory under `input_text/`, e.g., `input_text/my_project/`.

2. **Run extraction:**
   - From the project root, run:
     ```sh
     python -m core.extract
     ```
   - This will process the first `.txt` file found in any project subdirectory, generate outputs in `output_data/{project}/` with a timestamp.

3. **Run analysis:**
   - To analyze a specific output, run:
     ```sh
     python -m core.analyze output_data/{project}/{project}_YYYYMMDD_HHMMSS_graph.json --html
     ```
   - The analysis HTML report will open automatically in your browser.
   - For Markdown output:
     ```sh
     python -m core.analyze output_data/{project}/{project}_YYYYMMDD_HHMMSS_graph.json --output output_data/{project}/{project}_YYYYMMDD_HHMMSS_analysis.md
     ```

## Notes
- All outputs are timestamped for easy versioning.
- The analysis HTML report now opens automatically after generation.
- You can have multiple projects in `input_text/` and process them independently.

## Requirements

- Python 3.8+
- Required packages:
  - networkx
  - matplotlib
  - google-generativeai (for text extraction)
  - Other dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Google API key for Gemini (if using text extraction)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 