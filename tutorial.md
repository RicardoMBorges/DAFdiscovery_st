# 📘 DAFdiscovery Tutorial – How to Use the App

**DAFdiscovery** is an interactive application built with [Streamlit](https://streamlit.io/) to integrate NMR, Mass Spectrometry (MS), and Bioactivity data using the **STOCSY (Statistical Total Correlation Spectroscopy)** methodology. It helps reveal structure–activity relationships and correlations in metabolomics datasets.

🔗 Access the app here: [https://dafdiscovery.streamlit.app](https://dafdiscovery.streamlit.app)

---

## 📁 Required File Formats

Prepare your input files as follows:

| Data Type      | Format   | Description |
|----------------|----------|-------------|
| NMR (1D)       | `.csv`   | Rows = variables (ppm), columns = samples |
| MS (LC-MS, GC-MS) | `.csv`| Rows = features (m/z or RT), columns = samples |
| Bioactivity    | `.csv`   | Rows = bioassays, columns = samples |
| Metadata (optional) | `.csv` | Annotations about the samples (name, group, class, etc.) |

> ⚠️ Or any combination of two of those: NMR + MS; NMR + BioAct; MS + BioAct (this one is probably the most common use for Natural Products).
 
> ⚠️ Column names in all datasets must match exactly (they represent sample IDs).

> ⚠️ The metadata.csv file must: (1) be comma-separated files; (2) have as columns: "Samples"; "NMR_filename" if you have NMR; "MS_filename" if you have MS; "BioAct_filename" if you have BioActivity results. And the strings must contain the same text as their filename counterparts in the NMR.csv file, the MS.csv file, or the BioAct.csv file.

---

## How to Use DAFdiscovery

### 1. Launch the app

Open the web app in your browser:  
👉 [https://dafdiscovery.streamlit.app](https://dafdiscovery.streamlit.app)

### 2. Upload your files

- **NMR Data**: upload your `.csv` file containing NMR intensities
- **MS Data**: upload your `.csv` file with MS feature intensities
- **Bioactivity Data**: upload your `.csv` file with bioassay results
- (Optional) **Metadata**: upload your `.csv` file with sample annotations

> 📌 All sample columns must be named identically across all files.

---

### 3. Choose the analysis mode

After uploading:

- **Automatic with Bioactivity**: use bioactivity data as the *driver* for STOCSY.
- **Manual selection**: choose a specific NMR signal (ppm) or MS feature (m/z) as driver.

---

### 4. Select a correlation model (STOCSY mode)

You can apply different correlation transformations:

| Model         | Recommended for |
|---------------|-----------------|
| `linear`      | Simple Pearson correlation |
| `exponential` | Decay-type relationships |
| `sinusoidal`  | Cyclic data patterns (e.g., circadian rhythms) |
| `sigmoid`     | Dose–response or saturation patterns |
| `gaussian`    | Peak-like correlation (e.g., chromatographic profiles) |

---

### 5. View the results

- The app will generate interactive plots based on your data.
- Correlations are color-coded: **red = positive**, **blue = negative**.
- You’ll also get a DataFrame with **correlation** and **covariance** values.

---

### 6. Export the results

You can download:

- 📄 Correlation table (`.csv`)
- Plots in `.pdf` and interactive `.html` formats
- Interactive figures (with hover support for MS/NMR info)

---

## Example Workflow

You have:

- 1D NMR data from plant extracts
- Bioactivity data from antibacterial assays

You want to find which NMR signals correlate with the observed bioactivity.

The app will:

1. Calculate correlations between NMR signals and bioactivity values
2. Display a STOCSY-style NMR plot with color-coded correlations
3. Provide data files for further interpretation or publication

---

## ❓ FAQ

**Do I need to upload all data types (NMR, MS, Bioactivity)?**  
No. You can use just two (e.g., NMR + Bioactivity), or even one (e.g., NMR only for manual analysis).

**Can I use NMR as a driver?**  
Yes. You can choose any specific ppm value as the driver.

**Can I use MS as a driver?**  
Yes. Choose any m/z value as the driver.


**How can I choose the BioActivity as a driver?**  
You don´t. If you have BioActivity as part of the imported information in your metadata, it will calculate the correlations from the BioAct "signal" into the NMR and/or MS data.

**Does it work with PCA or PLS-DA results?**  
STOCSY is an independent analysis but exported results can complement PCA or PLS-DA interpretations.

---

## 👨‍💻 About the Project

DAFdiscovery is developed and maintained by Ricardo M. Borges and collaborators as part of research in natural products chemistry, metabolomics, and analytical spectroscopy.

📧 Contact: [ricardo.borges@ufrj.br](mailto:ricardo.borges@ufrj.br)  
🌱 Linked to IPPN-UFRJ and LabMAS

---

## 🧬 Contributing

Want to improve the app or report an issue?

1. Fork this repository
2. Create a branch (`feature/your-feature-name`)
3. Submit a pull request with a clear description

---

🧠 *Open science for data-driven discovery.*
