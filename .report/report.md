GRACE
Graph-based Robust Anomaly detection with Contrastive Encoding
Bao cao nghien cuu: Phuong phap phong thu truoc Mimicry Attacks trong Prov-HIDS

Dua tren paper: "Sometimes, You Aren't What You Do"
Goyal et al., NDSS 2023

 
1. Tong Quan Paper Goc

1.1 Boi Canh Va Van De
Cac he thong phat hien xam nhap dua tren provenance graph (Prov-HIDS) la the he IDS moi nhat, phan tich do thi nhan qua de phat hien APT attacks. Thay vi nhin tung hanh vi don le, Prov-HIDS nhin toan bo chuoi nhan qua de phat hien bat thuong.
Tuy nhien, paper NDSS 2023 cua Goyal et al. chung minh rang tat ca cac he thong Prov-HIDS hien tai deu co the bi qua mat 100% thong qua mimicry attacks - tan cong gia mao hanh vi binh thuong.

1.2 Cac He Thong Duoc Nghien Cuu
He thong	Phuong phap	Diem yeu chinh	Gadget tan cong
StreamSpot	K-hop neighborhood, unweighted sum	Global aggregation bi pha loang	G1
Unicorn	Histogram phan phoi, HistoSketch	Normalized distribution bi bao qua mat	G2
ProvDetector	Top-K path selection, Doc2Vec + LOF	Downsampling bi chiem cho	G3
Pagoda	Path anomaly score, 2 lop detect	Ca 2 lop cung 1 diem yeu	G1 + G2
FGA (GCN)	Graph autoencoder, mean pooling	Mean pooling bi dilute	G2

1.3 Ba Mimicry Gadgets
Gadget 1 - Unweighted Edge Injection
Co che: Them nhieu benign substructures bat ky vao attack graph. Khong quan tam ti le, chi can du nhieu la embedding bi pha loang.
Target: StreamSpot, Pagoda (lop 1)
Chi phi: 250-300K edges tren StreamSpot
Ket qua: 100% evasion rate

Gadget 2 - Distributional Mimicry
Co che: Inject edges theo dung ti le phan phoi syscall cua benign graph. Phai nghien cuu ki tan suat de inject dung ty le.
Target: Unicorn, FGA, Pagoda (lop 2)
Chi phi: 40-100K edges
Ket qua: 100% evasion rate

Gadget 3 - Path Flooding / Downsampling Bypass
Co che: Inject cac benign paths co regularity score thap hon tat ca attack paths, chiem het top-K slots. Attack paths bi day xuong duoi khong duoc encode.
Target: ProvDetector
Chi phi: Chi can 9K edges - rat thap
Ket qua: 100% evasion rate

Sequential Attack G3 -> G2
Ket hop G3 thay doi structure truoc, sau do G2 thay doi distribution. Hai chieu perturbation phoi hop tao trajectory trong embedding space ma cac he thong khong chong duoc.
Ket qua voi baseline: 100% evasion, chi can 25-49K edges

Ket luan chinh cua paper goc
• Tat ca 5 Prov-HIDS bi evade 100% - moi he thong co it nhat 1 gadget pha duoc hoan toan
• Nguyen nhan goc re: bounded depth, global aggregation, va downsampling
• Paper de xuat: phat hien o granularity nho hon (node/edge level) se robust hon
• Open-source benchmark: https://bitbucket.org/sts-lab/mimicry-provenance-generator

 
2. Kien Truc GRACE

2.1 Dong Luc Thiet Ke
GRACE (Graph-based Robust Anomaly detection with Contrastive Encoding) duoc de xuat de giai quyet chinh xac 3 diem yeu ma paper goc chi ra:

Diem yeu trong paper goc	Gadget khai thac	GRACE giai quyet bang
Bounded depth beta - khong nhin duoc causal chain dai	G1, G2, G3	Graphormer global attention - khong co gioi han depth
Global aggregation bi dilute	G1, G2	Node-level scoring - score tung node doc lap
Downsampling mat attack paths	G3	Khong co downsampling - encode tat ca nodes

2.2 Pipeline Tong The
GRACE xu ly provenance graph qua 4 buoc chinh:

Pipeline GRACE
Buoc 1 - Data Augmentation: Tao 2 views tu cung 1 graph bang DropEdge + MaskFeature
Buoc 2 - Structural Encoding: Tinh Centrality, SPD, Edge Type cho tung node
Buoc 3 - Graphormer Encoder: Global self-attention tren toan bo graph
Buoc 4 - Node Scoring Space: InfoNCE loss de hoc separation, anomaly score tung node

2.3 Ba Loai Structural Encoding
Centrality Encoding
Moi node duoc gan 1 vector 128-dim dua tren in-degree va out-degree cua no trong toan graph:
h_centrality = Emb_table[in_degree] + Emb_table[out_degree]
Khi G2 inject 10K benign nodes, degree cua Firefox attack node van giu nguyen - centrality encoding van nhan ra no la hub bat thuong.

Shortest Path Distance (SPD) Encoding
SPD la so hops ngan nhat giua moi cap nodes trong toan graph. SPD duoc dung lam bias trong attention matrix:
Attention(i,j) = (Q_i dot K_j) / sqrt(d) + b_SPD(i,j) + b_edge_type(i,j)
Du G2 inject 10K nodes vao giua, SPD(Firefox, server) = 2 van khong doi. Graphormer van biet Firefox lien quan truc tiep den server doc hai.
Khac voi Edge Type: SPD la khoang cach (bao nhieu buoc), Edge Type la noi dung (lam gi tren duong di). Hai thong tin doc lap bo sung cho nhau.

Edge Type Encoding
Moi loai system call (read, write, exec, connect...) co embedding rieng. Edge exec tu Firefox den /clean la bat thuong du xung quanh co nhieu read edges.
G2 inject nhieu read edges khong the che giau exec edge cua attack - edge type encoding van nhan ra.

2.4 InfoNCE Contrastive Learning
GRACE chi train tren benign graphs - khong co attack graphs trong training. Day la unsupervised/self-supervised approach.
InfoNCE tao 2 views tu cung 1 benign graph bang augmentation khac nhau, sau do:
•	Positive pair: Z1[node_i] vs Z2[node_i] - cung node, khac view -> keo gan lai
•	Negative pairs: Z1[node_i] vs Z2[node_j] - khac node -> day xa nhau

Sau training, attack nodes co embedding xa benign cluster -> anomaly score cao -> detect.

2.5 Node-Level Scoring vs Graph-Level
Tieu chi	Baseline (graph-level)	GRACE (node-level)
Don vi phan tich	1 vector cho ca graph	1 vector cho tung node
Bi dilute boi G1/G2?	Co - them nhieu nodes lam pha loang	Khong - tung node score doc lap
Bi bypass boi G3?	Co - mat paths bi downsampling	Khong - encode tat ca nodes
Complexity	O(E) den O(N*d^2)	O(N^2 * d) - nang nhat

Tai sao node-level manh hon graph-level
Graph-level: Firefox + 10000 benign nodes -> 1 vector trung binh -> Firefox bi chim
Node-level: Firefox co 1 vector rieng -> anomaly score cao -> khong bi anh huong
Tuong tu: Max pooling manh hon mean pooling voi mimicry attacks

