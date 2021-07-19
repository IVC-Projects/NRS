## Neural Reference Synthesis for Inter Frame Coding
DANdan Ding*, Xiang Gao*, Chenran Tang*, Zhan Ma**<br>
\* Hangzhou Normal University<br>
** Visionular Inc.<br>
——————

Introduction:
We propose to jointly optimize these two submodules to effectively exploit the spatiotemporal correlations for better characterization of structural and texture variations of pixel blocks. Specifically, we develop two deep neural networks
(DNNs) based models, called EnhNet and GenNet, for reconstruction enhancement and reference generation respectively. The EnhNet model is mainly leveraging the spatial correlations
within the current frame, and the GenNet is then augmented by further exploring the temporal correlations across multiple frames. Moreover, we devise a collaborative training strategy in
these two neural models for practically avoiding the data over-fitting induced by iterative filtering propagated across temporal reference frames.
