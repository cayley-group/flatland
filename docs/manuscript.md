Flatland: A simulation environment for simplifying the development of polymer structure prediction tools

Christopher Beitel


ABSTRACT

The development and debugging of modern neural network approaches for protein structure prediction has the potential to be a somewhat complicated task. Here we provide a framework, Flatland, for simulating training datasets of the form used by current state-of-the-art systems. The central task performed is to evolve a population of polymer sequences, predict their folded structure, produce distance matrix and bond angle tensors, and calculate spectra of per-polymer compound or other polymer interaction likelihoods (emulating protein-protein and compound-protein interaction datasets). This framework intentionally performs such simulations in a highly simplified manner that does not accurately model the real biological processes. Relatedly, it provides the means to simplify the problem to an arbitrary extent - including permitting the reduction from three to two dimensions, reducing the amino acid alphabet from twenty-one to three, and simplifying characteristics of the evolutionary optimization. We provide the framework together with a collection of starting-point notebooks for training neural network models to learn from such data by way of an open source repository to kick-start the development efforts of those interested in contributing to solving this vitally important problem.


INTRODUCTION

Fully solving the protein folding problem will confer the ability to design cures for many diseases in silico and can reasonably be expected to significantly reduce the time it takes to bring a new drug to market. This has relevance to not only classically critical diseases such as cancer and neurodegenerative diseases but more recently (as of 2021) the rapid design and development of vaccines. These methods will be enhanced by the means to predict static structures of proteins that can be reasonably said to have these; even greater value is expected from being able to further predict an accurate dynamical model that can be used to design, for example, small molecules to induce desired conformational changes in the protein. Further, medical science will be advanced by the accumulation of a repository of such structures for all known protein sequences (e.g. beginning with all human and common animal model proteins). This might be compared to existing broad efforts to obtain reference quality genome sequences for many of the species on earth - so too are needed reference quality protein structures for all of the same. Further in regard to scale, in silico protein design efforts will be able to explore a larger space of protein designs proportionate to the time it takes to perform each simulation so computation time has a direct relationship to the kind of human value that can be produced.

Naturally, such a broad challenge is one to be solved by the broader human research community and need not be rendered feasible for a single group. Here we begin to explore a strategy for doing so by seeking to broadly stimulate the development efforts of the community by lowering the barrier to entry for parties interested in working on the problem. One barrier to working on the problem we hope to help overcome is the educational one and to do so by providing thoroughly clear notebook narratives that help new developers get started.

The larger barrier we hope to help overcome is one common to the development of neural network models - that being the difficulty of debugging them due to their often high complexity. We will do so through two avenues, one being by providing a simplified simulation environment that can be used to produce data of the same structure used by current state-of-the art methods. Currently this includes being able to produce evolutionary histories, residue-residue distance matrices, and solved structures. Debugging and development will be facilitated along this dimension by virtue of the reduced complexity of simulations and the control developers are provided by being able to modify simulation parameters (towards their further understanding model behavior). The second avenue will be to provide simple baseline models to operate on this data that are themselves tested (as modules and at the system level) so as to simplify the process of developing model improvements.

Lastly, we aim to include in the simulation environment the means to generate additional forms of data that may be relevant to advancing beyond the current state-of-the-art. Currently, SOTA methods may approximate experimental performance for the determination of static crystal structures but are reported to not perform as well for structures that are not easily crystallizable. Further, methods capable of predicting the structure of multimers and the physics of complex protein-protein and compound-protein interactions are still considered future-generation. To this end, we are inspired to explore the possibility of co-training structure solving models on the addition of compound-protein and protein-protein (collectively “affinity” or “structure-encoding”) data. Concisely, one can imagine the structure of a protein is to some extent encoded in the spectrum of interaction affinities between it and a panel of compounds or other proteins (a bit like a hologram). To cultivate the exploration of such methods we thus aim to include the simulation of such data types in our system. We further seek for the inclusion of this extension to illustrate one means by which the community may explore the potential of new data types in silico towards understanding their usefulness and better predicting the related resource requirements to produce the datasets and computational complexity of using them in training models.


DESIGN

The general architecture of data and network components used in SOTA structure solvers is summarized in Figure 1. 


Figure 1. The general structure of many SOTA structure solvers and required data. 

From this we extracted the following core requirements for our simulator:

Evolution. Each polymer sequence belongs to an evolutionary history wherein the function of folded polymer forms determined individual fitness thereby imbuing that history with patterns expected to be useful for subsequently guessing at the tertiary structure of folded polymers.
Structures. Solved structures are produced for each input polymer. These structures are repeatable and stable (thereby contributing to the potential of models to be trained to predict that consistent “answer”).
Distances and angles. Convenience methods should be provided to obtain distances and angles from structures in a standardized manner including to avoid an opportunity for user error.
Affinity spectra. Spectra of polymer-compound interactions should be produced to enable subsequent exploration of the usefulness of such data types.


METHODS

Molecular dynamics simulation

Simulations of molecular dynamics were performed using Jax MD. This included simulations according to an energy function consisting of parts designed to (simply) model covalent and weaker electrostatic interactions. The former is a functional description of spring-like behavior, [1], and is plotted for intuitive convenience in Figure 2.

V(r) =(r-r0)4 - (r-r0)2  					[1]


Figure 2. Visualization of spring-like bond energy function.

Hydrogen bonds are modelled according to [2.1, 2.2] and visualized for convenience in Figure 3.

f(r)  = 12k(r-r0)2 -D0,                    r <r0				[2.1]   
g(r)  = D0(e-2(r-r0) -2e-(r-r0)),     r r0				[2.2]


Figure 3. Visualization of electrostatic bond energy functions.

Disulfide and ionic bonds were not modelled in this iteration. Molecular dynamics simulations performed using Jax MD are those that are “probable” with respect to the energy functions given [1,2] yet not guaranteed to reach global energy minima. We look forward to work in the near future to include water molecules in simulations - as is a natural process by way of which protein structure emerges.

Structure solving

Minimum energy structures were solved by performing rounds of FIRE descent minimization over 10,000 steps each starting from initial conformations generated by perturbing an initial straight linear conformation with 100 steps of Brownian motion at 500 kT. While consistent minima were reached the consistency of structures fulfilling those minima was low. Thus we consider this useful for our purposes initially from an engineering perspective and anticipate improvements in structure consistency going forward. We anticipate that adding short side-chains will increase the complexity of possible structures and thus perhaps the uniqueness of energy minima.



Figure 4. Example of a minimum energy structure and a derived residue-residue distance matrix.

We acknowledge the potential of alternative methods of finding minimum energy conformations other than gradient methods from random initializations such as re-starting from perturbations of existing minima and/or genetic recombination of best known minima (including perturbed back-off from these) to create initializations. Lastly we acknowledge the potential interest of simulating polymers along a changing temperature gradient, beginning at zero temperature, that can effectively simulate polymers folding as they emerge from a pore.

Compound interaction profiling

Spectra of the energy of polymer-compound interactions were computed by adding each compound to the simulation domain of each “folded” polymer and continuing FIRE descent minimization further for 1,000 steps. For our purposes we simply considered the strength of the polymer-compound interaction to be the total reduction in energy between the folded polymer and (folded polymer + compound) conformations, summarized in [3].

I(p, c) = 1||E(p)||( Emin(p) - Emin(p + c) )					[3]



Figure 5. Example interaction simulation and spectrum.

Here for the sake of discussion we acknowledge the potential of alternative approaches such as computing summary statistics in the style of [3] over a large batch of Brownian simulations. We invite the reader to imagine how to improve these methods for measuring interaction strength in a manner that would sufficiently simulate compound-protein interaction for the sake of aiding early development.

Evolutionary optimization of polymer sequences

We performed evolutionary optimization of randomly initialized polymer sequences for the purpose of minimally simulating the kind of data used by SOTA structure inference algorithms. This includes an evolutionary history for each sequence that can be mined to inform guesses about macro structure. Initially, for the sake of simplicity, this evolution was performed with a fitness function given as the minimum energy of each folded polymer computed as described above (“Structure solving”). While it might be more interesting to evolve a population with high affinity for a spectrum of compounds this task would take approximately num_compounds*optimization_rounds longer so we leave it for near-subsequent work.

Polymer populations were evolved with simple point mutation at a base mutation frequency of 0.15 with a population size of 5,000 and 200,000 for each primary dataset (Table 1), respectively. To improve stability, mutation rates were reduced linearly according to the distance of the average population fitness from the maximum achievable population fitness. Populations at each generation were produced as simple weighted random samples according to the fitness scores of the previous. Fitness scores were re-scaled in a relative fashion to their position on a scale between the minimum and maximum population fitness.

Polymers consisted of elements from an alphabet of three instead of 21 with positive, negative, and neutral elements in regard to their affinity for water molecules. Here, polymer elements are composite representations of various atoms including oxygen and hydrogen groups involved in hydrogen bonding, hydrophobic or hydrophilic side chains that interact differently with water, and N and C termini polypeptide bonding atoms.

DATASETS

Two datasets generated using the Flatland environment are made available for download both to reduce the barrier to getting started and reduce the need for re-computation. These datasets are those used by the demonstration notebooks we provide in which we show these can be used to train simplified models. All datasets are provided via a public requester-pays Google Cloud Storage bucket with relevant access instructions provided in our GitHub repository.

Dataset name
num_examples
Data size
Num. cpd.
Download URL
flatland-public-small
5,000
NN Gb
100
gs://flatland-public/small
flatland-public-xlarge
2,000,000
NN Gb
1,000
gs://flatland-public/xlarge

Table 1. Public datasets. Metadata and links for two public datasets are provided (a small and an extra-large version). Metadata includes the size of each dataset as well as the number of compounds included in the simulation of polymer/compound interaction profiling.

We provide an exploratory notebook to illustrate the above methodological concerns of polymer and compound interaction simulation as well as evolution which further includes the exact configurations used to generate the datasets given in Table 1. The notebook is available to be used without additional setup by way of the Google Collaboratory environment, https://colab.research.google.com/github/cayley-group/flatland/blob/master/nb/data-generation.ipynb.

EXPERIMENTS

Initially we used data generated by the Flatland environment to train a model for the sake of simply beginning to close the loop between our intended use and generated data. Near-subsequent work will extend the structure solver with latent representations derived from simulated evolutionary histories as is standard in SOTA methods.

Learning to fold

Here we sought to learn a structure optimizer that when provided the coordinates of one structure would tend to produce a second structure of lower energy. A graph neural network was initialized with one node per polymer element and once edge joining each of all pairs of elements. At each step, node features were simply the 2D coordinate of that node. Node functions (that aggregate information from neighbor node and edge representations and update node representations) were learned by gradient backpropagation from a loss function that simply computed the difference between the energies of the input and output structures (and incentivized this to be reduced). Almost unnecessarily, [3].

L(si,sj) = E(si) - E(sj)					[3]	

For aggregate energy functions (over all bond and interaction types), E, and input and output structures si and sj respectively. This can be viewed as meta-optimization, i.e. optimizing to find a better optimizer that is tailored to this particular data context. One might anticipate some improvement would come from familiarity optimizing certain kinds of structures while other benefits may perhaps come from learning behaviors that have the tendency to escape local minima.

For a model trained with MM hyperparameters, after NN steps the evaluation loss was E. The demonstration notebook can be used via https://colab.research.google.com/github/cayley-group/flatland/blob/master/nb/structure-solver.ipynb. We are looking forward to the potential that such an optimizer will further improve by being provided evolution-derived representations and that the optimizer can become further domain-specialized to navigate local minima using such representations as a guide (indeed without the need to tune the exact means for doing so as was a free variable in the approaches of Yang et al., 2020 and Senior et al., 2019).

DISCUSSION

Here we presented the Flatland simulation environment as a means for generating simplified simulation datasets that can be used to enable the early stage development of novel protein structure prediction algorithms. We provide the data, code, as well as demonstration models openly for others to build upon. In the future, these resources will hopefully not only enable early-stage development but also be a useful educational resource (such as for training workshops). Further, we anticipate that as the level of realism of these simulations is improved they may be used as a mechanism of end-to-end pre-training prior to training on more expensive (and smaller) experimentally-generated datasets.


ADDITIONAL INFORMATION

Code and documentation (including demonstration notebooks) are made available via github.com/cayley-group/flatland as well as in a permanent archived form via Zenodo at doi.here.1234. We invite the community to participate in the development of this resource by filing or commenting on a GitHub issue at github.com/cayley-group/flatland/issues.


REFERENCES

J Yang, I Anishchenko, H Park, Z Peng, S Ovchinnikov, D Baker, Improved protein structure prediction using predicted interresidue orientations, PNAS, 117: 1496-1503 (2020).

Senior, Andrew W., et al. "Improved protein structure prediction using potentials from deep learning." Nature 577.7792 (2020): 706-710.

Schoenholz, Samuel S., and Ekin D. Cubuk. "Jax md: End-to-end differentiable, hardware accelerated, molecular dynamics in pure python." (2019).

Drori, Iddo, et al. "Accurate Protein Structure Prediction by Embeddings and Deep Learning Representations." arXiv preprint arXiv:1911.05531 (2019).

Wang, Zhaohui et al. “An array of 60,000 antibodies for proteome-scale antibody generation and target discovery.” Science advances vol. 6,11 eaax2271. 11 Mar. 2020, doi:10.1126/sciadv.aax2271

Luck, Katja et al. “A reference map of the human binary protein interactome.” Nature vol. 580,7803 (2020): 402-408. doi:10.1038/s41586-020-2188-x

Abbott, Edwin A. (1884). Flatland: A Romance in Many Dimensions. New York: Dover Thrift Edition (1992 unabridged). p. ii.
