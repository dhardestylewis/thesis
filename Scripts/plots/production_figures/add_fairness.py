import sys

pth = r'Thesis_Draft\Draft_v1\Lewis_2026_NIMBYism_Austin_Thesis.tex'
with open(pth, 'r', encoding='utf-8') as f:
    text = f.read()

fairness_block = r"""
\newpage
\subsection{Fairness and Subgroup Evaluation (FNR Gaps)}
\label{app:fairness_evaluation}

To calculate the False Negative Rate (FNR) gaps:
\begin{itemize}
    \item \textbf{Geography:} Subgroups defined by the 10 Austin City Council Districts.
    \item \textbf{Socioeconomic Vulnerability:} Subgroups defined by the UT-Austin Uprooted Displacement Risk index (Vulnerable, Active, Chronic).\footnote{Categories follow the 2024 City of Austin update. The original 2018 Uprooted report used a six-category gentrification typology (Susceptible, Early Type 1, Early Type 2, Dynamic, Late, Continued Loss) based on Bates (2013).}
    \item \textbf{Classification Threshold:} To empirically ground the spatial error analysis and prevent misleading impressions caused by severe class imbalance, the binary prediction threshold for FNR evaluation is strictly anchored to the empirical background protest rate ($\mu_{y}$). Any localized zoning project with a predicted opposition probability exceeding the citywide historical baseline is classified as a predicted positive case, ensuring the algorithm's errors are evaluated against actual class incidence rather than arbitrary percentiles.
\end{itemize}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{Figures/ch4/fig_ch4_22_spatial_error.png}
    \caption[Geographic Error Distribution]{\textbf{Geographic Error Distribution.} Spatial distribution of classification errors across Austin's 10 City Council districts, measuring structural variance between spatial false positive (predicting a petition that never arrived) and false negative occurrences.}
    \label{fig:spatial_error}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{Figures/ch4/fig_ch4_32_fpr_fnr_longitudinal.pdf}
    \caption[Longitudinal FNR Disparities]{\textbf{Longitudinal Disparity and Displacement Risk.} Tracking of False Negative Rate (FNR) gaps structured mapping subgroup severity defined out of the UT-Austin Uprooted Displacement index.}
    \label{fig:fnr_longitudinal}
\end{figure}

\subsection{Temporal Drift Threshold Sensitivity}
"""

text = text.replace(r"\subsection{Temporal Drift Threshold Sensitivity}", fairness_block)

with open(pth, 'w', encoding='utf-8') as f:
    f.write(text)

print("SUCCESS")
