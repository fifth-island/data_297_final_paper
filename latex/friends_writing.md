
\documentclass[9pt,a4paper,twocolumn,twoside]{rho-class/rho}
\usepackage[english]{babel}

%----------------------------------------------------------
% Title
%----------------------------------------------------------

\title{Dual-Channel Contrastive Encoding of Sperm Whale Coda}

%----------------------------------------------------------
% Authors, Affiliations and dates
%----------------------------------------------------------

\author{João Quintanilha}
\author{Emma Virnelli}

\affil{Tufts University}

\dates{May 8, 2026}

%----------------------------------------------------------
% Corresponding author information
%----------------------------------------------------------

\corres{\textsuperscript{\textasteriskcentered}Corresponding author: \href{mailto:joao.quintanilha@tufts.edu}{joao.quintanilha@tufts.edu} (J. Quintanilha).}
\corres{\textsuperscript{\textdagger}These authors contributed equally to this work.}

%----------------------------------------------------------
% Abstract
%----------------------------------------------------------

\begin{abstract}
    Sperm whales (Physeter macrocephalus) communicate using short sequences of clicks known as codas. While prior work has focused on coda type classification, individual whale identification remains challenging due to data scarcity and the absence of supervised training signals. We introduce DCCE (Dual-Channel Contrastive Encoder), a self-supervised architecture that respects the biological decomposition of coda communication into rhythm and spectral channels. The model uses a GRU to encode inter-click interval (ICI) sequences and a CNN to process mel-spectrograms, then jointly embeds both representations into a 64-dimensional space. Using cross-channel positive pairs, DCCE learns without explicit labels.
\end{abstract}

%----------------------------------------------------------
% Keywords (if we want to
%----------------------------------------------------------

\keywords{Sperm Whale, Contrastive Learning}

%----------------------------------------------------------

\begin{document}

    \maketitle
    \thispagestyle{firststyle}

%----------------------------------------------------------
% Your content goes here in two columns
%----------------------------------------------------------

\section{Introduction}
Sperm whales (Physeter macrocephalus) are among the most well-studied species in the field of communication, owing to their high intelligence and complex social structures. Traditionally, sperm whale codas have been analysed by grouping them into different types based on click count and inter-click intervals (ICIs) (Beguš et al., 2026). Building on Project CETI's foundation, the present study tests whether biological inductive bias can outperform raw data scale. Sperm whales are a compelling target for this question: they possess the largest brain of any known species, live in multigenerational matrilineal families with documented cultural transmission, and produce click sequences called "codas": short, rhythmic patterns that, much like human dialects, vary across populations (Project CETI).
\begin{figure}[H]
    \centering
    \includegraphics[width=\columnwidth]{figures/coda (1).png}
    \label{fig:coda (1)}
\end{figure}

Recent work has revealed that these codas carry two independent information channels encoded in the same acoustic signal:

\begin{itemize}
    \item A rhythm channel (the timing pattern of clicks) that encodes what type of coda it is.
    \item A spectral channel (the acoustic texture within each click) that encodes who is speaking.
\end{itemize}


No existing model utilizes both channels together by design. Our research question is: 
\textbf{Can a purpose-built encoder respecting the whale's two-channel decomposition (rhythm for coda type, spectral for identity) outperform a generalist audio model trained at scale?}

\subsection{Related Work}
Current research has paved the way for the state of sperm whale communication today. The foundation for this research began in 2016, when there was an established foundational unit and coda type classification, being able to identify the certain type of clicks released (Gero et al., 2016). Later on, after identifying it, it was realized that codas have a combinatorial structure: when the feature combines with already known elements like rhythm and tempo, the sperm whales are able to produce a larger set of distinct signals than were previously thought, around 10 times larger (Sharma et al., 2024). Building on this distinction comes a formalized spectral texture as a second independent axis of coda structure, essentially adding a "vowel-like" channel that operates separately from rhythm and timing, further supporting communication being multi-layered rather than a single undifferentiated signal (Beguš et al., 2024). Beyond this, there is a theoretical basis for why these two channels, rhythm and spectral, can be treated as independent, providing the field a principled reason to analyze them separately and confirming they carry distinct information (Leitão et al., 2023). All of these developments lead us to the current state, which is WhAM, Whale Acoustic Model, produced by Project CETI: the first large generative model for cetacean audio, trained on around 10,000 codas, marking a significant methodological leap where data and theory were mature enough to support large-scale generative modeling rather than just classification and labeling (Paradise et al., 2025). Working through where the state of sperm whale communication research is, there is a prominent gap: there is \textbf{
no prior work explicitly encodes the rhythm/spectral decomposition as a representation learning architecture
}. All of this research ties to what we are currently looking at, being able to define how sperm whales communicate and how we decided to separate ourselves by combining these two channels rather than keeping them separate.

\section{Data}

The study started with 1,501 DSWP audio recordings from the Dominica Sperm Whale Project (2005–2010, all EC1 clan). After removing 118 noise-contaminated files, 1,383 clean codas remained spanning 3 social units (A, D, F) and 22 coda types. A further 621 codas lacked individual identification (IDN=0, primarily from Unit F), leaving 762 identified codas from 12 unique whales. Labels (social unit, coda type, individual ID) were reconstructed from DominicaCodas.csv (Sharma et al., 2024), where codaNUM2018 maps exactly to DSWP file indices. Severe class imbalance (Unit F = 59.4\% of codas; IDN=0 = 44.9\%) motivated the use of macro-F1 as the primary metric, weighted random sampling during training, and balanced class weights for linear probes.


\subsection{Data Identification}
Two main characteristics define a sperm whale coda: the number of clicks and the (ICI). The coda type naming system follow Gero et al., where the first number indicates the total click count, and the letter signifies the timing pattern: R for regular spacing (e.g., 5R), D for decreasing intervals (e.g., 4D), i for increasing intervals, and "+" for a long pause between groups (e.g., 1+1+3).

\subsection{Data Challenge}
The core challenge was that within the dataset of 1,501 coda recordings (22 coda types, 3 social units), none of them had a label: no social group, no coda type, no individual whale ID, and no timing sequences between clicks. Thus, a cascading problem formed. Without labels, the team was not able to train supervised models, create positive pairs for contrastive learning, establish an evaluation protocol, or design linear probes. The temporal problem comes into play as well. Because the three social units were recorded in different epochs spanning 2005 to 2010, a model risks learning when a recording was made rather than anything meaningful about the whales themselves. The final problem is the size differentiation between the social units, with Unit F carrying more than half of the data at around 59\%, meaning a naive model could simply predict Unit F and appear accurate.

To solve these issues, respectively, the team manually assembled labels from public sources, used Macro-F1 scoring to treat all classes equally regardless of size, and thoroughly cleaned any abrupt sound variation, resulting in a smaller but higher-quality dataset.

\begin{figure}[H]
    \centering
    \includegraphics[width=\columnwidth]{figures/data downsizing.png}
    \label{fig:data downsizing}
\end{figure}
\section{Methodology}

\section{Results}

\section{Discussion}

\section{Conclusion}

%----------------------------------------------------------

\printbibliography



%----------------------------------------------------------

\end{document}