# Architecture Ablation Study

This appendix reports an ablation study of the main components of GPRM. We compare the complete model, denoted as **All**, against seven variants in which one component is removed at a time: CGRA coordinates, GIN-based pattern encoding, graph neural network encoding, Laplacian positional encodings, mobility information, placement ordering, and transformer input scaling.

![Architecture ablation study of GPRM.](gprm_ablation.png)

**Figure 1: Ablation study of GPRM.** Comparison between the complete model and variants without CGRA coordinates, GIN-based pattern encoding, graph neural network encoding, Laplacian positional encodings, mobility information, placement ordering, and transformer input scaling. WO denotes without.

Figure 1 shows the training and evaluation curves for all variants. Overall, the complete model provides the most consistent behavior across all metrics, achieving low training loss, high training accuracy, and the best valid mapping rate.

Most ablated variants remain close to the complete model in terms of mapped nodes per mapping, indicating that they can still learn placement decisions. However, their consistently lower valid mapping rates indicate that completing the full mapping correctly requires the joint contribution of the complete set of encodings.

The most significant degradation occurs when removing the placement ordering information, presenting higher loss, lower accuracy, and a substantially lower valid mapping rate. This indicates that the placement sequence provides an important signal for making sequential placements.

Finally, it is important to state that the removal of scaling can degrade the stability of the model's learning for larger numbers of parameters in internal experiments.