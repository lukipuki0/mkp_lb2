

<!-- Start of picture text -->
Soft&<br><!-- End of picture text -->



# **Artificial Bee Colony with Momentum-Guided Search and Starfish Exploration for Complex and Multimodal Problems** 

## **Harun Akbulut**<sup>***1**</sup> 

*1 Department of Artificial Intelligence and Machine Learning, Faculty of Computer and Information Sciences, Kayseri University, Türkiye aharunakbulut@kayseri.edu.tr **ABSTRACT** The Artificial Bee Colony (ABC) algorithm is a widely used swarm intelligence method due to its simple structure and strong exploitation capability. However, it may experience performance loss in complex and multi-modal optimization problems due to its limited exploration capability and tendency toward early convergence. In this study, an optimization algorithm called Momentum-Guided Search and Starfish Exploration-based Artificial Bee Colony (MSE-ABC) is proposed to overcome these limitations. The proposed method preserves the observer and scout bee phases employed in the classical ABC algorithm while enhancing its global search capability with trigonometric exploration operators inspired by starfish optimization algorithm (SFOA). Furthermore, a momentum-based steering mechanism and dynamic exponential step size control are integrated to accelerate convergence during the search process and adaptively maintain the exploration-exploitation balance. The performance of the MSE-ABC algorithm was comprehensively evaluated through CEC 2022 benchmark tests, a five different real-world engineering design optimization problems, and deep learning hyperparameter optimization experiments conducted on the CIFAR-10 datasets. Experimental results demonstrate that the proposed method achieves superior or competitive performance when compared with eight state-of-the-art metaheuristic algorithms. The statistical significance of the obtained results was verified using the Wilcoxon signed-rank test and the Friedman test. The findings reveal that the MSE-ABC algorithm offers an effective and reliable optimization approach for complex and multi-modal optimization problems. **Keywords** : Artificial Bee Colony, Starfish-Inspired Optimization, Hybrid Metaheuristic Optimization, Complex and Multimodal Optimization, Hyperparameter Optimization **1. Introduction** Optimization algorithms are numerical methods developed to find optimal solutions to complex problems encountered in a wide range of disciplines such as engineering, economics, logistics, data science, and finance, or to approximate these solutions as closely as possible [1]. These algorithms aim to optimize an objective function defined under certain constraints and are often designed to perform effectively in complex, multimodal, and nonlinear search spaces [2]. Today, the increasing scale of problems, the growing number of decision variables, and the increasingly complex structure of search spaces significantly complicate the optimization process. In this context, classical deterministic and derivative-based optimization approaches, such as gradient descent and Newton-based methods, often fall short in many real-world problems and frequently get stuck in local optimum solutions due to their reliance on strong mathematical assumptions such as continuity, differentiability, and convexity [3]. The loss of effectiveness of such methods in the face of noisy, discontinuous, or multi-modal objective functions has increased the need for alternative search strategies. To overcome these limitations, heuristic and metaheuristic optimization algorithms inspired by biological, physical, and social processes in nature have attracted considerable interest and been intensively researched over the past thirty years [4]. Metaheuristic algorithms offer more flexible and applicable alternatives compared to deterministic methods due to their ability to obtain solutions close to the global 



optimum within reasonable computation times in large, complex search spaces, without requiring rigid assumptions about the problem structure or derivative information [5]. For this reason, meta-heuristic algorithms are widely applied not only in engineering and optimization problems, but also in a wide variety of fields such as healthcare, defense, finance, energy, transportation, telecommunications, and social sciences, with the aim of producing effective and efficient solutions [6-18]. 

In the literature, meta-heuristic algorithms are generally examined under two main categories: single-solution-based algorithms, which rely on the iterative improvement of a single solution candidate, and population-based algorithms, which are based on the simultaneous evolution of a population consisting of multiple candidate solutions [19]. Single-solution-based methods such as Simulated Annealing and Tabu Search stand out for their strong local search capabilities; population-based algorithms, on the other hand, can produce more effective results in multi-modal problems thanks to their advantage of being able to explore different regions of the solution space simultaneously. Swarm intelligence methods, which occupy an important place among population-based metaheuristic algorithms, perform searches by modeling the collective and cooperative behavior of agents such as particles, bees, ants, or flocks, aiming to naturally maintain the balance between exploration and exploitation [20]. Particle Swarm Optimization (PSO) [21], Ant Colony Optimization (ACO) [22], and Artificial Bee Colony (ABC) algorithm [23], are among the most widely used swarm intelligence-based approaches in the literature. Real-world problems in modern engineering and science are often complex, multimodal, nonlinear, and NP-hard, making them very difficult to solve using traditional mathematical methods. While current meta-heuristic algorithms offer nature-inspired mechanisms to overcome this complexity, they face serious limitations such as early convergence, falling into local optimum traps, and an imbalance between exploration and exploitation capabilities in the search space. Although there are currently more than 540 meta-heuristic algorithms, the fact that, according to the “No Free Lunch” theorem, no algorithm can perform optimally on all problem types necessitates the continuous development of new methods [24].  Although the Artificial Bee Colony (ABC) algorithm stands out due to its simple structure and low parameter requirements, its standard version has low exploitation capacity because it updates only one dimension when generating new solutions [25].  To address these weaknesses, researchers are modifying the mechanisms of existing algorithms, particularly by integrating the information of globally optimal or elite individuals into the solution search equations [26]. Among the improved mechanisms are the adaptation of coefficients that guide the search process and the addition of modification rates that determine how many dimensions of the solution will be changed [27]. Furthermore, approaches that utilize distance information between individuals, rather than focusing solely on fitness values, increase convergence speed while preserving population diversity [28]. To prevent algorithms from becoming stagnant, feeding the scout bee phase from historical experience or opposition-based learning, rather than making it completely random, is a critical improvement. Such mechanical changes make the ABC algorithm more resilient in complex optimization landscapes [29]. As a result, replacing one-dimensional updates with multidimensional and guided strategies is essential to meet modern optimization needs. Enhancing the exploitation capabilities of these population-based methods enables the production of high-precision engineering solutions. Researchers continue to develop new approaches that dynamically adjust algorithm parameters to overcome these limitations. 

Modified and improved models of the ABC algorithm focus on strengthening the algorithm's weak local search capabilities and increasing convergence speed. For example, the ABC-MNG (Multi-Neighbor Guidance) model balances by selecting not only the best neighbor for each individual but also the farthest and closest neighbors based on Euclidean distance. In this model, 



population diversity is effectively preserved by using the cosine similarity metric to capture the directional information of candidate solutions [30]. Another important development, the MGABC approach, guides the search equations by establishing an elite group consisting of the top 10% of the population [31]. The ASRGABC model offers a mechanism that adaptively selects the appropriate search strategy by monitoring success rates in current and previous iterations. In this model, the population is randomly divided into groups, and scout bees are made to follow local leaders to avoid local optima [32]. The IABC model, designed for gridbased path planning problems, incorporates a path straightening strategy that eliminates unnecessary corners in the path. IABC also enhances global performance by incorporating a local search mechanism that optimizes the best solution at the end of each iteration [33]. In the ABCG variant, inspired by the gravity model, the attraction force between individuals is calculated, and neighbor selection is performed accordingly. In the improved models, the scout bee phase scans opposite regions using adversarial learning instead of generating completely random solutions. In variants developed for test scenario generation, the fitness function evaluates values in a four-level categorical structure. This categorization distinguishes between borderline-valid, valid, borderline-invalid, and invalid values, thereby improving exploitation quality [34]. Some advanced versions adaptively change the population size throughout iterations to minimize computational cost. Exploitation-focused speed equations that use the information of the global best individual (gbest) are frequently integrated into the local search phase of ABC in such models. These mechanical improvements enable ABC to produce accurate results in complex physical and engineering problems [35]. Neighborhood topologies and hierarchical leadership systems balance the algorithm's exploration capability with exploitation. Dynamic perturbation rates adjust the scanning sensitivity in the solution space on a problem-by-problem basis. As a result, the improved models exhibit much faster convergence performance than standard ABC. A hybrid method is an integrated approach that aims to overcome the shortcomings of a single algorithm by combining the advantages of two or more meta-heuristic algorithms or different computational techniques. Among hybridization methods, it is common to blend one algorithm's global exploration capability with another's local exploitation power or to use machine learning for parameter tuning. The HABC-QL model combines the ABC algorithm with Q-learning to enable the adaptive selection of the optimal neighborhood search operator during the scout bee phase [36]. In a model used for gas pipeline optimization, the ABC algorithm is hybridized with Deep Reinforcement Learning (Actor-Critic), where a data-driven policy determines which dimensions to update. In this approach, the Actor network calculates update probabilities, while the Critic network evaluates the quality of these decisions to manage the algorithm [37]. The HABCSMO model combines ABC's exploration capability with the hierarchical leader-based learning mechanisms of the Spider Monkey Optimization (SMO) algorithm [29]. Similarly, the ABC-CSO (Chicken Swarm Optimization) hybrid maximizes energy efficiency in wireless networks by integrating chicken swarm behavior into ABC's cluster head selection process [38]. Another hybrid, ABC-SA, incorporates the Simulated Annealing temperature control mechanism into the optimization process, allowing initially poor solutions to be accepted. This enables the algorithm to escape local optima traps and explore a broader global search space [39]. Differential Evolution (DE) operators or Particle Swarm Optimization (PSO) velocity equations are also used in hybrid models to enhance exploitation capability [29]. Chaotic maps are incorporated into hybrid structures to make the initial population's randomness more balanced. Tabu Search (TS) mechanisms are used in these systems to prevent the repetition of previously searched regions. Reinforcement learning-based structures enable the algorithm to update its own strategy during operation. Function approximation methods improve solution accuracy in complex engineering problems involving continuous variables. As a result, hybrid structures create much more robust systems by 



disciplining the random nature of metaheuristics with scientific rigor. These integrated approaches are considered among the most powerful tools in modern optimization. 

**<u>Table 1.</u>** <u>ABC algorithm and its variants</u> 

|**Algorithm**|**Year**|**Inspiration**<br>**Source**|**Main**<br>**Modification**|**Advantages**|**Limitations**|
|---|---|---|---|---|---|
|**Gbest-guided**<br>**ABC (GABC)** **[40]**|**2010**|ABC + global-<br>best guidance|Global-best term<br>added to search<br>equation|Faster<br>convergence;<br>stronger<br>exploitation|Reduced diversity;<br>premature<br>convergence risk|
|**Modified ABC**<br>**(MABC)** **[15]**<br>|**2010**<br>|ABC with<br>modified<br>neighbor<br>generation<br>|New<br>perturbation<br>strategy<br>|Improved local<br>search; better<br>robustness<br>|Slightly higher<br>complexity<br>f|
|**PS-ABC (Hybrid**<br>**ABC–PSO) [41]**<br>|**2015**<br>|Bees + particle<br>swarm<br>dynamics<br>|PSO velocity<br>concept<br>embedded into<br>ABC<br>|Faster<br>convergence;<br>improved<br>exploitation<br>o|Parameter tuning<br>required<br>|
|**hABCDE (Hybrid**<br>**ABC with**<br>**Differential**<br>**Evolution) [42]**<br>|**2017**<br>|ABC + DE<br>operators<br>|DE mutation<br>integrated into<br>ABC phases<br>-|Enhanced<br>convergence<br>accuracy;<br>balanced search<br>pr|Higher<br>computational cost<br>|
|**ABCADE**<br>**(Adaptive DE-**<br>**based ABC) [43]**<br>|**2017**<br>|ABC + adaptive<br>differential<br>operators<br>|Adaptive DE<br>mutation<br>strategies<br>re|Stronger<br>exploitation;<br>faster<br>convergence<br>|Extra parameters<br>|
|**sdABC (Self-**<br>**adaptive**<br>**Differential ABC)**<br>**[44]**<br>|**2019**<br>|ABC + self-<br>adaptive<br>differential<br>search<br>al|Multiple DE<br>strategies with<br>adaptation<br>|Improved<br>robustness;<br>reduced<br>stagnation<br>|Algorithmic<br>overhead<br>|
|**Bayesian**<br>**Estimation ABC**<br>**(BEABC) [45]**<br>|**2022**<br>|ABC +<br>Bayesian<br>learning<br>n|Probabilistic<br>parameter<br>estimation<br>|High accuracy;<br>strong learning<br>capability<br>|Computationally<br>expensive<br>|
|**Modified ABC for**<br>**Classification**<br>**Optimization [46]**<br>|**2022**<br>u|ABC +<br>classification<br>objective<br>|Task-oriented<br>modification<br>|Improved<br>classification<br>accuracy<br>|Problem-dependent<br>|
|**Modified ABC**<br>**with DE (mABC-**<br>**DE) [47]**<br>J|**2022**<br>|ABC + DE<br>mutation &<br>crossover<br>|Hybrid variation<br>operators<br>|Fast<br>convergence;<br>strong global<br>|Increased<br>complexity<br>|
|<br>||||<br>search<br>||
|Table 1 presents t<br>purpose of the tabl<br>the literature to ove<br>weak exploitation<br>that early ABC im<br>studies show a shif<br>Table 1 reveals a c<br>modifications towa<br>developing new hy<br>avoiding excessive|he imp<br>e is to<br>rcome<br>capabil<br>provem<br>t towar<br>lear tr<br>rds hy<br>brid A<br>compl|ortant variants<br>systematically<br>the known limit<br>ity and prematu<br>ents focused p<br>ds hybridizatio<br>end in the devel<br>brid and adaptiv<br>BC-based optim<br>exity.|and hybrid appr<br>summarize the f<br>ations of the cla<br>re convergence.<br>rimarily on enh<br>n, adaptive, and l<br>opment of ABC<br>e models. This<br>ization methods|oaches of the A<br>undamental app<br>ssical ABC algor<br>An examinatio<br>ancing exploitati<br>earning-based m<br>algorithms, mo<br>trend provides s<br>that achieve hig|BC algorithm. The<br>roaches proposed in<br>ithm, particularly its<br>n of Table 1 reveals<br>on capability. Later<br>echanisms. Overall,<br>ving from structural<br>trong motivation for<br>h performance while|







<!-- Start of picture text -->
45 x10° Hybrid Optimization Algorithms in Computer Science (ScienceDirect)<br>o<br>12<br>2)<br><<br>2<br>S<br>2<br>=<br>a<br>-_<br>°2<br>=<br>s4<br>z<br>3 a2<br>0 G = 5 2 : - ~ ~ if<br>2000 2005 2010 2015 2020 2025<br>Year<br><!-- End of picture text -->



imbalance even in hybrid structures and prevents algorithms from reaching their full potential. Therefore, there is a need for a new hybrid approach that integrates the strengths of one algorithm to compensate for the weaknesses of another, while also incorporating an intelligent mechanism to manage the evolving nature of the search over time. 

This study aims to fill precisely this methodological gap. The robust exploitation structure of the ABC algorithm and the superior exploration power of SFOA are brought together for the first time in this research in a complementary synergy. The proposed Momentum-Guided Search and Starfish Exploration-based Artificial Bee Colony (MSE-ABC) algorithm takes this combination beyond a mechanical addition by enhancing it with a “momentum” mechanism that learns from past search knowledge and dynamically manages the exploration-exploitation transition. Thus, while SFOA's wide-angle exploration effectively scans the solution space in early iterations, ABC's focused exploitation is accelerated in later stages by the directed information provided by momentum, enabling more stable and precise convergence. The original value of this work stems from the first-time integration of two algorithms into a unified strategy, the idea of providing memory-based guidance to exploration operators, and the introduction of an adaptive framework that dynamically controls search balance. The effectiveness of MSE-ABC has been demonstrated through comprehensive comparative tests and real-world problems, establishing it as a new and powerful option for complex optimization tasks. The main contributions of this study to the literature can be listed as follows: • A directed and adaptive hybrid optimization model (MSE-ABC) has been proposed, which integrates the ABC and SFOA algorithms under a single umbrella for the first time. • SFOA's powerful trigonometric exploration operators based on sine and cosine are systematically integrated with a momentum-based mechanism that directly utilizes past search information. • The exploration-exploitation balance is dynamically and adaptively adjusted based on iterations, thereby developing a directed search strategy that differs from mechanical hybrid approaches based solely on operator combination. • The proposed MSE-ABC algorithm is designed to reduce early convergence and local optimum getting stuck problems, exhibiting a more stable convergence behavior, especially in complex and multi-modal search spaces. • The performance of the method has been comprehensively tested on CEC 2022 benchmark functions, three different real-world engineering design problems, and deep neural network hyperparameter optimization scenarios. • The MSE-ABC algorithm has been compared with contemporary and powerful metaheuristic algorithms such as success history based differential evolution (SHADE) [49], dream optimization algorithm (DOA) [50], covariance matrix adaptation evolution strategy (CMAES) [51], SFOA, standard ABC, and ABC based on adaptive search strategy and random grouping mechanism ASRGABC [52]. 

- The statistical significance of the obtained results has been verified using the Wilcoxon signed-rank test. 

The remaining sections of the article are organized as follows: Section 2 introduces the basic ABC and SFOA algorithms and presents the detailed structure and mathematical formulation of the proposed MSE-ABC hybrid optimization method. Section 3 contains experimental results, comparisons, and statistical analyses performed on CEC 2022 [53] benchmark functions, engineering design problems, and deep neural network hyperparameter optimization. Section 4 summarizes the overall results of the study and discusses possible future research directions. 



## **2. Materials and methods** 

This section provides a detailed examination of the fundamental principles and algorithmic mechanisms of the ABC algorithm and SFOA. The search strategies of both methods, their effects on the exploration–exploitation balance, and their behavior patterns in the solution space are analyzed within a theoretical framework. Furthermore, the step-by-step operation of the algorithms is presented through pseudo-code representations. In this context, the conceptual structure, design motivation, and implementation steps of the MSE-ABC algorithm, a new hybrid approach developed by leveraging the complementary strengths of ABC and SFOA, are explained in detail. **2.1. Artificial Bee Colony Optimization Algorithm** The ABC algorithm is a population-based optimization method that mimics the collective foraging and information sharing behaviors exhibited by honeybees in nature. In the ABC algorithm, each possible solution represents a food source around the beehive, and the quality of this source represents the solution's fitness value. The algorithm consists of three main bee groups: worker bees, scout bees, and observer bees. Worker bees attempt to improve the solution by performing local searches around already discovered food sources. This process is modeled by applying a random but guided perturbation within the neighborhood structure of the current solution. Observer bees, on the other hand, probabilistically select food sources with higher nectar quantities based on the information shared by worker bees in the dance area. This mechanism enhances exploitation capability by enabling more intensive exploration of promising regions in the solution space. Scout bees, on the other hand, abandon solutions that cannot be improved upon after a certain number of iterations, generating entirely random new points in the solution space. This gives the algorithm a powerful exploration mechanism that prevents it from getting stuck in local minima. One of the most notable features of the ABC algorithm is that the step size automatically decreases based on progress in the solution space. As solutions get closer to each other, the effect of perturbations diminishes, and the search process becomes more precise. This allows the algorithm to behave adaptively without requiring an additional control parameter. However, the relatively weak local exploitation in the classical ABC structure can cause the convergence speed to be insufficient for some problems. This limitation has paved the way for the emergence of many improved and hybrid approaches based on ABC in the literature. **2.2. Starfish Optimization Algorithm** The SFOA is a modern meta-heuristic optimization method inspired by the trigonometric and symmetrical movements of starfish in response to environmental stimuli. The fundamental motivation behind SFOA is to create a dynamic search behavior that can both perform wideangle scanning in the solution space and conduct precise searches in promising regions. In the algorithm, each individual represents a sea star located in the solution space, and the movements of these individuals are modeled using sine and cosine functions. 

Observer bees, on the other hand, probabilistically select food sources with higher nectar quantities based on the information shared by worker bees in the dance area. This mechanism enhances exploitation capability by enabling more intensive exploration of promising regions in the solution space. Scout bees, on the other hand, abandon solutions that cannot be improved upon after a certain number of iterations, generating entirely random new points in the solution space. This gives the algorithm a powerful exploration mechanism that prevents it from getting stuck in local minima. 

The search process of SFOA is guided by the relative positions of the individuals to the current best solution. Thanks to trigonometric functions, individuals can exhibit nonlinear and periodic movements, jumping to different regions of the solution space. This allows the algorithm to gain strong exploration capabilities in its early stages. In subsequent iterations, as the amplitude of movement gradually decreases, individuals converge around the best solution and shift towards exploitation behavior. 



One of the most significant advantages of SFOA is its ability to establish a smooth and continuous balance between the search and exploitation phases, rather than a sharp transition between them. This structure facilitates avoiding local minima, particularly in multi-modal and complex optimization problems. Furthermore, the simplicity of the algorithm's mathematical structure enhances its applicability and computational efficiency. However, the lack of sufficiently sharp local search capability in the pure SFOA structure for some problems makes hybridization approaches attractive. 

**2.3. Proposed Momentum-Guided Search and Starfish Exploration-based Artificial Bee Colony (MSE-ABC)** The MSE-ABC algorithm is a new hybrid optimization approach that integrates the phase-based search architecture of the ABC algorithm with the directed and trigonometric motion mechanisms of SFOA under a momentum-supported update strategy. Although the ABC algorithm offers strong local search capabilities through worker, scout, and explorer bee phases, it is known that its global exploration ability weakens over time, especially in complex and multimodal optimization problems. In contrast, SFOA exhibits a broader and more directed exploration behavior in the solution space thanks to its structure inspired by the radial and multiarmed movements of starfish. In the proposed MSE-ABC algorithm, the basic phase architecture of ABC is preserved; however, the classical neighborhood generation mechanisms have been restructured with directed search components derived from SFOA. Specifically, SFOA's exploration strategies, which involve trigonometric guidance around the global best solution and exploration through randomly selected dimensions, have been integrated into the initial phase of the algorithm to enhance ABC's exploration capability in early iterations. Additionally, the multi-pronged approach used in SFOA's exploitation behavior has been incorporated into ABC's worker bee phase, strengthening its local improvement capacity. One of the fundamental challenges encountered in the hybridization process is harmoniously combining the different search dynamics of two algorithms. To this end, a momentum-based update mechanism has been defined in the MSE-ABC algorithm. The momentum term, derived from the difference between the global best solutions in successive iterations, provides directional information to the search process and prevents erratic oscillations. This approach ensures that the search direction is maintained in a stable manner, particularly during the exploitation phase. However, it has been observed that managing the exploration–exploitation balance with fixed parameters leads to suboptimal results in different iteration stages. Therefore, in the MSE-ABC algorithm, an adaptive step size that decreases as the iteration progresses is used to gradually shift the exploration behavior towards exploitation. While the solution space is broadly scanned with large steps in the initial phase, smaller steps are used in subsequent iterations to perform precise local searches. Thanks to this dynamic structure, both the algorithm's global search capability and convergence speed have been significantly increased. 

In the MSE-ABC algorithm, each candidate solution is defined as an nD-dimensional vector as in Eq. (1). 

Xi = [ x𝑖,1, xi,2, … , xi,nD] (1) 

Here, 𝑋𝑖 is the nD-dimensional decision variable vector representing the 𝑖<sup>𝑡ℎ</sup> candidate solution. 𝑥{𝑖,𝑗}, corresponds to the 𝑗<sup>𝑡ℎ</sup> decision variable component of the 𝑖<sup>𝑡ℎ</sup> solution, and 𝑛𝐷 denotes the dimension of the optimization problem. 



The initial population is randomly distributed within the feasible search space using a uniform distribution, as expressed in Eq. (2). 



Here, 𝑙𝑏𝑗 and 𝑢𝑏𝑗 denote the lower and upper bounds of the j-th decision variable, respectively. 

𝑟𝑎𝑛𝑑(0,1) is a uniformly distributed random number in the interval [0,1]. This formulation ensures a diverse and unbiased initial population. 

Once the population is initialized, the fitness value of each candidate solution is evaluated using the objective function𝑓(·), as shown in Eq. (3). 𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑖 = 𝑓( 𝑋𝑖) (3) Here, 𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑖 represents the fitness value of the 𝑖<sup>𝑡ℎ</sup> solution, and 𝑓(·) denotes the objective function to be minimized. At the initialization stage, the global best solution is determined based on the minimum fitness value among all individuals in the population, as defined in Eqs. (4) and (5). 𝐵𝑒𝑠𝑡𝐹𝑖𝑡(0) = min (4) 𝑖=1,2…,𝑁𝑝𝑜𝑝<sup>𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑖</sup> Here, 𝐵𝑒𝑠𝑡𝐹𝑖𝑡(0) denotes the minimum fitness value obtained at the initial iteration, 𝐵𝑒𝑠𝑡𝑃𝑜𝑠(0) represents the corresponding decision vector, 𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑖 is the fitness value of the 𝑖<sup>𝑡ℎ</sup> candidate solution, and 𝑁𝑝𝑜𝑝 indicates the population size. These quantities define the reference solution that guides the subsequent search process. 𝐵𝑒𝑠𝑡𝑃𝑜𝑠(0) = 𝑎𝑟𝑔𝑚𝑖𝑛𝑋𝑖𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑖 (5) Here, 𝐵𝑒𝑠𝑡𝐹𝑖𝑡(0) denotes the minimum fitness value obtained at the initial iteration, and 𝐵𝑒𝑠𝑡𝑃𝑜𝑠(0) represents the corresponding decision vector. This global best solution serves as a reference point for guiding the subsequent search process. To introduce directional memory into the search process, MSE-ABC explicitly defines a momentum term derived from successive global best solutions, which is calculated using Eq. (6). 𝑀𝑜𝑚𝑒𝑛𝑡𝑢𝑚(𝑡) = 𝐵𝑒𝑠𝑡𝑃𝑜𝑠(𝑡) − 𝐵𝑒𝑠𝑡𝑃𝑜𝑠(𝑡−1) (6) Here, 𝑀𝑜𝑚𝑒𝑛𝑡𝑢𝑚(𝑡) represents the displacement vector between the global best positions obtained at iterations 𝑡 and 𝑡−1. This term captures the dominant search direction in the solution space. The momentum contribution is scaled by a control parameter to regulate its influence on the search dynamics, as defined in Eq. (7). 

𝑀(𝑡) = 𝛽 × 𝑀𝑜𝑚𝑒𝑛𝑡𝑢𝑚(𝑡) (7) Here, 𝛽∈(0,1) is the momentum factor controlling the strength of directional persistence. Higher values of β emphasize exploitation along the dominant direction, whereas lower values increase exploratory behavior. 

In MSE-ABC, an adaptive step size mechanism is employed to dynamically balance exploration and exploitation during the optimization process, as formulated in Eq. (8). 





Here, 𝑠𝑡𝑒𝑝𝑖𝑛𝑖𝑡 and 𝑠𝑡𝑒𝑝𝑓𝑖𝑛𝑎𝑙 denote the initial and final step sizes, respectively, 𝑡 is the current iteration index, and 𝑀𝑎𝑥𝐼𝑡 represents the maximum number of iterations. This exponentially decreasing function enables wide-ranging exploration in early iterations and fine-grained local search in later stages. 

At each iteration, MSE-ABC probabilistically determines whether a candidate solution performs exploration or exploitation according to Eq. (9). 



Here, 𝑃𝑖 denotes the probability of selecting the 𝑖<sup>𝑡ℎ</sup> solution. This normalized formulation favors high-quality solutions while maintaining population diversity. 

To prevent stagnation, solutions that fail to improve for a predefined number of trials are abandoned and replaced, as expressed in Eq. (14). 





This equation represents the scout bee mechanism, where abandoned solutions are reinitialized randomly within the search space, thereby enhancing exploration and avoiding premature convergence. 

At the end of each iteration, the global best solution is updated as described in Eq. (15). 

𝐵𝑒𝑠𝑡𝐹𝑖𝑡(𝑡) = min( 𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑖(𝑡)) 



𝐵𝑒𝑠𝑡𝑃𝑜𝑠(𝑡) = 𝑎𝑟𝑔𝑚𝑖𝑛𝑋𝑖  ( 𝐹𝑖𝑡𝑛𝑒𝑠𝑠𝑖(𝑡) ) 

These expressions update the best fitness value and its corresponding solution vector obtained up to iteration 𝑡. 

Algorithm 1 presents the pseudocode of the proposed MSE-ABC algorithm. The proposed MSE-ABC algorithm integrates the guided exploration capability of the Starfish Optimization Algorithm with the strong local exploitation mechanism of the Artificial Bee Colony algorithm. At the beginning, a population of candidate solutions is randomly initialized and evaluated. During each iteration, a starfish-inspired phase is first employed to perform momentum-guided exploration or global-best-oriented exploitation, which enhances population diversity and prevents premature convergence. Subsequently, the employed and onlooker bee phases of the ABC algorithm are applied to intensify the search around promising regions through fitnessbased local exploitation. Finally, a scout bee mechanism is used to reinitialize stagnated solutions, maintaining diversity throughout the optimization process. The global best solution is updated iteratively, and the convergence behavior is recorded until the termination criterion is satisfied. By combining the complementary strengths of SFOA and ABC, the proposed MSE-ABC algorithm achieves a balanced exploration–exploitation trade-off. The momentum-guided starfish phase accelerates convergence toward high-quality regions, while the ABC-based local search phases refine solutions and improve stability, particularly in complex and multimodal optimization problems. Algorithm 1. Pseudocode of the Proposed MSE-ABC (Momentum-Guided Search and - <u>Starfish Exploration based Artificial Bee Colony) Algorithm</u> **Algorithm 1** _MSE-ABC: Momentum-Guided Search and Starfish Exploration-based Artificial Bee Colony_ **Input:** f(·), lb, ub, nD, Npop, MaxIt, β (momentum_factor), GP, stepinit, stepfinal, limit, fobj **Output:** BestPos, BestFit, Curve **INITIALISATION** 1 **for** i = 1 **to** Npop **do** 2 **for** j = 1 **to** nD **do** 3  xi,j ← lbj + rand(0,1) × (ubj − lbj) **[Eq. 2]** 4 **end for** 5  Fitnessi ← fobj(Xi) **[Eq. 3]** 6  triali ← 0 _// stagnation counter for scout bee_ 7 **end for** 8  BestFit(0) ← mini(Fitnessi) **[Eq. 4]** 9  BestPos(0) ← argminXi(Fitnessi) **[Eq. 5]** 10  BestPosprev ← BestPos(0) _// store for momentum computation_ **MAIN LOOP** 11 **for** t = 1 **to** MaxIt **do** 12 _// Adaptive step size (exponential decay)_ 13  step(t) ← stepinit × (stepfinal / stepinit)^(t / MaxIt) **[Eq. 8]** 14 _// Momentum vector from successive global bests_ 15  Momentum(t) ← BestPos(t−1) − BestPosprev **[Eq. 6]** 16  M(t) ← β × Momentum(t) **[Eq. 7]** 17  BestPosprev ← BestPos(t−1) _// update for next iteration_ _<u>// ═══════════ STARFISH PHASE ═══════════</u>_ 

By combining the complementary strengths of SFOA and ABC, the proposed MSE-ABC algorithm achieves a balanced exploration–exploitation trade-off. The momentum-guided starfish phase accelerates convergence toward high-quality regions, while the ABC-based local search phases refine solutions and improve stability, particularly in complex and multimodal optimization problems. 



18 **for** i = 1 **to** Npop **do** 19  ri(t) ← rand(0,1) 20 **if** ri(t) < GP **then** _// GP = 0.5; exploration branch_ 21  j ← randi(nD) _// single random dimension_ 22  r1 ← rand(0,1) 23  x<sup>new</sup> i,j ← xi,j + step(t) × [r1 × (BestPosj − xi,j) + β × Momentumj(t)] **[Eq. 10]** 24 **else** _// exploitation branch_ 25  k ← randi(Npop), k ≠ i 26  r2 ← rand(0,1) 27  X<sup>new</sup> i ← Xi + r2 × (BestPos − Xk) **[Eq. 11]** 28 **end if** 29 _// Boundary control: clipping to [lb, ub]_ 30 **for** j = 1 **to** nD **do** 31 **if** x<sup>new</sup> i,j < lbj **or** x<sup>new</sup> i,j > ubj **then** 32  x<sup>new</sup> i,j ← lbj + rand(0,1) × (ubj − lbj) 33 **end if** 34 **end for** 35 _// Greedy selection_ 36 **if** fobj(X<sup>new</sup> i) < Fitnessi **then** 37  Xi ← X<sup>new</sup> i;  Fitnessi ← fobj(X<sup>new</sup> i);  triali ← 0 38 **else** 39  triali ← triali + 1 40 **end if** 41 **end for** _// ═══════════ ABC: EMPLOYED BEE PHASE ═══════════_ 42 **for** i = 1 **to** FoodNumber **do** 43  k ← randi(FoodNumber), k ≠ i 44  j ← randi(nD) _// single random dimension_ 45  φ ← (rand(0,1) − 0.5) × 2 _// φ_ ∈ _[−1, 1]_ 46  x<sup>new</sup> i,j ← xi,j + (xi,j − xk,j) × φ + step(t) × β × Momentumj(t) **[Eq. 12]** 47 _// Boundary control (same rule as lines 30–34)_ 48 _// Greedy selection_ 49 **if** fobj(X<sup>new</sup> i) < Fitnessi **then** 50  Xi ← X<sup>new</sup> i;  Fitnessi ← fobj(X<sup>new</sup> i);  triali ← 0 51 **else** 52  triali ← triali + 1 53 **end if** 54 **end for** _// ═══════════ ABC: ONLOOKER BEE PHASE ═══════════_ 55 **for** i = 1 **to** FoodNumber **do** 56  Pi ← 0.9 × [1 / (1 + Fitnessi)] / max[1 / (1 + Fitness)] + 0.1 **[Eq. 13]** 57 **end for** 58  tsel ← 0;  i ← 1 59 **while** tsel < FoodNumber **do** 60 **if** rand(0,1) < Pi **then** _// roulette-wheel acceptance_ 61  k ← randi(FoodNumber), k ≠ i 62  j ← randi(nD) 63  φ ← (rand(0,1) − 0.5) × 2 64  x<sup>new</sup> i,j ← xi,j + (xi,j − xk,j) × φ **[Eq. 12]** 65 _// Boundary control (same rule as lines 30–34)_ 66 **if** fobj(X<sup>new</sup> i) < Fitnessi **then** 67  Xi ← X<sup>new</sup> i;  Fitnessi ← fobj(X<sup>new</sup> i);  triali ← 0 68 **else** 69  triali ← triali + 1 70 **end if** 71  tsel ← tsel + 1 72 **end if** 73  i ← (i mod FoodNumber) + 1 _// cycle through food sources_ 74 **end while** _<u>// ═══════════ SCOUT BEE PHASE ═══════════</u>_ 



75  idx ← argmaxi(triali) 76 **if** trialidx ≥ limit **then** _// abandonment condition (limit = 100)_ 77 **for** j = 1 **to** nD **do** 78  xidx,j ← lbj + rand(0,1) × (ubj − lbj) **[Eq. 14]** 79 **end for** 80  Fitnessidx ← fobj(Xidx) **[Eq. 3]** 81  trialidx ← 0 _// reset counter after re-initialisation_ 82 **end if** _// ═══════════ GLOBAL BEST UPDATE ═══════════_ 83  [minFit, idx] ← mini(Fitnessi) 84 **if** minFit < BestFit **then** 85  BestFit ← minFit;  BestPos ← Xidx **[Eq. 15]** 86 **end if** 87  Curve(t) ← BestFit _// store convergence value_ 88 **end for** 89 **return** BestPos, BestFit, Curve **<u>Note:</u>** <u>The code will be shared on GitHub at</u> **<u>github/hakbulut60</u>** <u>for the benefit of the research community.</u> **3. Experimental Results** In this section, the effectiveness and robustness of the proposed MSE-ABC algorithm have been evaluated through a comprehensive experimental analysis. The experimental study first utilized standard CEC 2022 benchmark functions to demonstrate the algorithm's general search capability and convergence behavior in complex and multimodal optimization problems. In the second stage, experiments were conducted on the Welded Beam, Pressure Vessel, Tension Spring, Speed Reducer, and Three Bar Truss engineering design problems [54] to demonstrate the practical applicability and performance of the proposed method on real-world problems. Finally, deep learning-based hyperparameter tuning scenarios were addressed on the CIFAR10 datasets. All experiments were conducted in the MATLAB environment on a computer equipped with a Windows 10 Pro operating system, a 12th generation Intel® Core™ i5-12400 (2.50 GHz) processor, and 16 GB of RAM. All comparative experiments were conducted under common parameters to ensure a fair and reproducible evaluation; each algorithm was tested with 30 independent runs. In the experiments, the population size was set to 30, the maximum iteration count to 1000, and the algorithms included in the comparison were run with the default parameter values recommended in the literature. The proposed MSE-ABC algorithm was compared with current and widely used meta-heuristic algorithms such as ABC, ASRGABC, SFOA, SHADE, DOA, and CMAES. The ASRGABC algorithm was selected as the reference method because it is one of the most powerful variants of the ABC algorithm proposed in 2022 and has demonstrated superior performance in comparisons with other ABC derivatives in the literature. The other algorithms were selected because they have been recently proposed or are widely used in the literature. Performance evaluation was conducted based on the mean best value, standard deviation, computation time, convergence curves, and statistical significance tests (Wilcoxon signed-rank test and Friedman test). 

## **3.1. Performance Evaluation of MSE-ABC on the low-dimensional CEC 2022 Functions** 

In this subsection, the overall performance of the proposed MSE-ABC algorithm in multimodal optimization problems has been comprehensively evaluated using the CEC 2022 benchmark functions. The CEC 2022 test set, consisting of single-modal, simple multi-modal, hybrid, and composite functions, provides a challenging test environment that is widely accepted for balancing the exploration and exploitation capabilities of an optimization algorithm. The search space bounds for all CEC 2022 functions are defined in the range [−100, 100] as suggested in the literature, and the problem size is fixed at 20. 



**<u>Table 2.</u>** <u>Performance comparison of the algorithms on CEC 2022 benchmark functions.</u> 

|**Function**|**Algorithm**|**Best**|**Mean**|**SD**|**Time**<br>**Mean(s)**|<sup>**P Value**</sup>|**Effect**<br>**Size**|**Significant**|**Winner**|
|---|---|---|---|---|---|---|---|---|---|
||ABC|29560.8422|38959.3109|5860.3794|0.4193|1.7344E-06|0.9922|YES||
||ASRGABC|21270.6594|33879.7357|6928.9982|0.0739|1.7344E-06|0.9911|YES||
||SFOA|300.6294|325.1797|25.7670|0.0387|1.7344E-06|0.0703|YES||
|**CEC1**|SHADE|20200.0221|42593.5540|9872.5933|0.3292|1.7344E-06|0.9929|YES|MSE-ABC|
||DOA|378.2491|2908.1311|2088.4087|0.0437|1.7344E-06|0.8960|YES||
||CMAES|10367.0743|25610.3060|11085.7794|0.1242|1.7344E-06|0.9882|YES||
||MSEABC<br>|300.0016<br>|**302.3045**<br>|7.2961<br>|0.0664<br>|-<br>|-<br>|-<br>||
||ABC<br>|449.0906<br>|449.1047<br>|0.0139<br>|0.4226<br>|1.7344E-06<br>|0.0307<br>|YES<br>||
||ASRGABC<br>|435.0379<br>|447.9226<br>|2.9034<br>|0.0720<br>|4.07151E-05<br>|0.0282<br>|YES<br>||
||SFOA<br>|404.1468<br>|446.4041<br>|16.9163<br>|0.0377<br>|0.031603382<br>|0.0249<br>|YES<br>||
|**CEC2**|SHADE<br>|445.2198<br>|450.9459<br>|5.2190<br>|0.3288<br>|1.7344E-06<br>|0.0347<br>|YES<br>|MSE-ABC|
||DOA<br>|405.1705<br>|442.7604<br>|13.1148<br>|0.0448<br>|0.015658483<br>|0.0168<br>|YES<br>||
||CMAES<br>|444.9708<br>|447.7610<br>|1.9834<br>|0.1220<br>-|0.001197338<br>|0.0278<br>|YES<br>||
||MSEABC<br>|400.0000<br>|**435.3032**<br>|21.3905<br>|0.0661<br>|-<br>|-<br>|-<br>||
||ABC<br>|600.0480<br>|600.2532<br>|0.1226<br>|0.6032<br>|1.7344E-06<br>|0.0004<br>|YES<br>||
||ASRGABC<br>|600.0000<br>|600.0003<br>|0.0015<br>|0.1480<br>|0.018518975<br>|0.0000<br>|NO<br>||
||SFOA<br>|601.9006<br>|607.0188<br>|3.2830<br>|0.1190<br>|1.7344E-06<br>|0.0144<br>|YES<br>||
|**CEC3**|SHADE<br>|600.0000<br>|600.0043<br>|0.0213<br>|0.4139<br>|0.002765274<br>|0.0000<br>|NO<br>|MSEABC|
||DOA<br>|600.0105<br>|600.0269<br>|0.0183<br>|0.1287<br>|1.7344E-06<br>|0.0000<br>|YES<br>||
||CMAES<br>|600.0000<br>|600.0043<br>|0.0232<br>|0.2019<br>|0.005319684<br>|0.0000<br>|YES<br>||
||MSEABC<br>|600.0000<br>|**600.0001**<br>|0.0001<br>|0.2212<br>|-<br>|-<br>|-<br>||
||ABC<br>|906.5135<br>|926.9227<br>|7.9052<br>|0.4639<br>|1.7344E-06<br>|0.0681<br>|YES<br>||
||ASRGABC<br>|853.0042<br>|883.1080<br>|14.7437<br>|0.0962<br>|0.000771217<br>|0.0219<br>|YES<br>||
||SFOA<br>|822.3624<br>|875.3199<br>|23.7345<br>|0.0609<br>|0.047161747<br>|0.0132<br>|YES<br>||
|**CEC4**|SHADE<br>|831.3686<br>|854.1242<br>|11.7210<br>|0.3518<br>|0.028485956<br>|0.0112<br>|YES<br>|CMAES|
||DOA<br>|829.8491<br>|853.1770<br>|14.4363<br>|0.0668<br>|0.035008957<br>|0.0123<br>|YES<br>||
||CMAES<br>|806.9647<br>|**814.8580**<br>|4.7747<br>|0.1481<br>|1.7344E-06<br>|0.0566<br>|YES<br>||
||MSEABC|828.8552|863.7586|19.5963|0.1132|-|-|-||
||ABC|902.1856|926.9594|19.2001|0.4746|1.7344E-06|0.4355|YES||
||ASRGABC|1554.7776|2195.2087|336.4505|0.0992|3.18168E-06|0.2520|YES||
||SFOA|912.0958|1326.4995|470.6188|0.0626|0.004114031|0.1922|YES||
|**CEC5**|SHADE|1061.3874|1320.0283|213.3860|0.3543|0.000715703|0.1961|YES|CMAES|
||DOA|946.1703|1282.0351|335.7614|0.0677|0.000715703|0.2193|YES||
||CMAES|900.0000|**900.0000**|0.0000|0.1512|1.7344E-06|0.4519|YES||
||MSEABC|1152.6300|1642.1033|325.9786|0.1155|-|-|-||
|**CEC6**|ABC|477288.8810|3585220.9318|1755488.7529|0.4282|1.7344E-06|0.9995|YES|MSEABC|





||ASRGABC|3099.5136|139083.5565|206507.3948|0.0783|1.7344E-06|0.9867|YES||
|---|---|---|---|---|---|---|---|---|---|
||SFOA|1894.5503|3259.7724|4528.9512|0.0429|2.12664E-06|0.4320|YES||
||SHADE|4896.7145|157262.6500|155531.3279|0.3338|1.7344E-06|0.9882|YES||
||DOA|1825.8205|3401.1701|2326.5377|0.0488|2.60333E-06|0.4556|YES||
||CMAES|1909.9953|8579.3243|5740.6128|0.1270|1.7344E-06|0.7842|YES||
||MSEABC|1803.2288|**1851.4390**|36.3020|0.0760|-|-|-||
||ABC|2081.5650|2116.2709|11.6535|0.6532|1.7344E-06|0.0388|YES||
||ASRGABC|2022.9817|2039.9697|7.6210|0.1764|0.025637124|0.0029|YES||
||SFOA<br>|2035.1513<br>|2089.1688<br>|27.6614<br>|0.1387<br>|1.92092E-06<br>|0.0264<br>|YES<br>||
|**CEC7**|SHADE<br>DOA<br>|2024.5180<br>2021.1286<br>|2038.6141<br>2035.8091<br>|9.1028<br>16.1436<br>|0.4456<br>0.1470<br>|0.075213312<br>0.734325291<br>|0.0000<br>0.0000<br>f|NO<br>NO<br>|MSEABC|
||CMAES<br>|2026.6048<br>|2049.8058<br>|13.8859<br>|0.2300<br>|5.30699E-05<br>|0.0077<br>|YES<br>||
||MSEABC<br>|2021.4588<br>|**2034.0688**<br>|10.0305<br>|0.2681<br>|-<br>|-<br>|-<br>||
||ABC<br>|2235.9918<br>|2248.9938<br>|4.5873<br>|0.6658<br>|1.7344E-06<br>|0.0122<br>|YES<br>||
||ASRGABC<br>|2224.0854<br>|2226.3152<br>|1.4878<br>|0.1911<br>|1.7344E-06<br>|0.0021<br>|YES<br>||
||SFOA<br>|2228.1674<br>|2241.5809<br>|21.3234<br>|0.1531<br>-|1.7344E-06<br>|0.0089<br>|YES<br>||
|**CEC8**|SHADE<br>|2223.0151<br>|2224.4449<br>|0.8883<br>|0.4473<br>|1.7344E-06<br>|0.0013<br>|YES<br>|MSEABC|
||DOA<br>|2220.6138<br>|2225.2045<br>|21.7159<br>|0.1585<br>|0.065641143<br>|0.0000<br>|NO<br>||
||CMAES<br>|2222.8712<br>|2263.9043<br>|52.0678<br>|0.2416<br>|1.7344E-06<br>|0.0187<br>|YES<br>||
||MSEABC<br>|2220.8399<br>|**2221.5873**<br>|0.5407<br>|0.2971<br>|-<br>|-<br>|-<br>||
||ABC<br>|2480.8115<br>|2480.9634<br>|0.1656<br>|0.6334<br>|1.7344E-06<br>|0.0001<br>|YES<br>||
||ASRGABC<br>|2480.7840<br>|2481.1505<br>|0.5582<br>|0.1770<br>|1.7344E-06<br>|0.0001<br>|YES<br>||
||SFOA<br>|2480.7813<br>|2480.7824<br>|0.0042<br>|0.1412<br>|1.7344E-06<br>|0.0000<br>|YES<br>||
|**CEC9**|SHADE<br>|2480.7815<br>|2481.3476<br>|0.8149<br>|0.4334<br>|1.7344E-06<br>|0.0002<br>|YES<br>|MSEABC|
||DOA<br>|2480.7815<br>|2481.1074<br>|0.4799<br>|0.1547<br>|1.7344E-06<br>|0.0001<br>|YES<br>||
||CMAES<br>|2480.8183<br>|2480.8627<br>|0.0278<br>|0.2291<br>|1.7344E-06<br>|0.0000<br>|YES<br>||
||MSEABC<br>|2480.7813<br>|**2480.7813**<br>|0.0000<br>|0.2721<br>|-<br>|-<br>|-<br>||
||ABC<br>|2506.4545<br>|2526.7185<br>|15.6361<br>|0.5795<br>|0.000615641<br>|0.0237<br>|YES<br>||
||ASRGABC<br>|2400.1874<br>|**2417.8506**<br>|35.3259<br>|0.1465<br>|0.000331726<br>|0.0199<br>|YES<br>||
||SFOA|2500.6237|3718.0449|1475.2751|0.1123|9.31566E-06|0.3365|YES||
|**CEC10**|SHADE|2400.0625|2430.9010|57.9033|0.4054|0.014795424|0.0146|YES|ASRGABC|
||DOA|2400.3469|2450.7043|55.2889|0.1195|0.673279807|0.0000|NO||
||CMAES|2448.3427|2612.0387|174.0866|0.2014|0.000135948|0.0556|YES||
||MSEABC|2402.0205|2466.9317|71.2790|0.2163|-|-|-||
||ABC|2900.0917|2901.5252|3.0776|0.7309|0.008729668|0.0407|YES||
||ASRGABC|2603.6193|2885.0325|100.0151|0.2196|0.036826128|0.0352|YES||
|**CEC11**|SFOA|2901.4211|3003.1668|99.6251|0.1855|4.28569E-06|0.0732|YES|MSEABC|
||SHADE|2900.0000|2975.1419|98.1528|0.4795|0.001286631|0.0645|YES||





|DOA<br>2600.7132|2850.6470|113.5986<br>0.1902|0.110925665 0.0000 NO|
|---|---|---|---|
|CMAES<br>2900.0000|2906.6667|25.3708<br>0.2739|0.020671114 0.0424 YES|
|MSEABC<br>2600.0000|**2783.3600**|189.4979<br>0.3592|-<br>-<br>-|
|ABC<br>2939.8784|2949.5028|3.8802<br>0.7555|0.000962659 0.0018 YES|
|ASRGABC 2940.0363|2949.9607|4.7120<br>0.2389|0.000830707 0.0020 YES|
|SFOA<br>2938.4962|2961.7442|31.0225<br>0.2020|0.000189097 0.0059 YES|
|**CEC12**<br>SHADE<br>2937.0831|2944.5037|5.1128<br>0.4923|CMAES<br>0.958990172 0.0000 NO|
|DOA<br>2934.0085|2945.0136|6.6894<br>0.2164|0.585711569 0.0000 NO|
|CMAES<br>2930.8492<br>|**2937.8137**<br>|5.0613<br>0.2902<br>|0.000615641 0.0021 YES<br>|
|MSEABC<br>2933.0101<br>|2944.1318<br>|6.5805<br>0.3932<br>|-<br>-<br>-<br>|
|**MSEABC 8 winner. CMAES 3 winn**<br>SD: Standard Deviation<br>|**er. ASRGABC**<br>|**1 winner**<br>|o|
|Table 2 presents the c<br>algorithms on the CEC 2<br>mean value (Mean), stan<br>signed-rank test p-value,<br>obtained from 30 indepe<br>The Friedman test, app<br>algorithms, revealed the<br>Examination of the Wilc<br>that statistically significa<br>algorithm in 10 out of 12<br>of the examined metahe<br>problem structure plays a<br>The distribution of winne<br>proposed MSE-ABC alg<br>mean value in 8 out of 1<br>the winner position in<br>ASRGABC reached the<br>SFOA, SHADE, and DO<br>The average rank analysi<br>with the lowest average r<br>When examined on a fun<br>its competitors in terms o<br>a standard deviation of 7<br>SHADE: 42,593.55; AS<br>all of them lagged signif<br>between 0.89 and 0.99 in<br>In CEC2, MSE-ABC ma<br>this function, the standar<br>of its competitors such<br>situation indicates that al<br>this function, it remained<br>Jou|omparative<br>022 benchm<br>dard deviatio<br>effect size, st<br>ndent runs.<br>lied to deter<br>presence of<br>oxon signed-<br>nt differences<br>test functions<br>uristic algori<br>decisive role<br>r algorithms<br>orithm demo<br>2 functions (6<br>3 functions<br>winner posit<br>A algorithms<br>s also support<br>ank across th<br>ction basis, in<br>f both solutio<br>.30. It was ob<br>RGABC: 33,8<br>icantly behin<br>dicate that thi<br>intained the w<br>d deviation of<br>as SHADE<br>though MSE<br>relatively lim<br>rnal|performance results o<br>ark functions. The resu<br>n (SD), mean executi<br>atistical significance,<br>mine the overall perf<br>statistically significant<br>rank test results used f<br>at the p < 0.05 level<br>(83.3%). This finding<br>thms differ markedly<br>in algorithm performa<br>was determined based o<br>nstrated its overall su<br>6.7%). This was follo<br>(25.0%), namely CE<br>ion in 1 function (8.<br>did not achieve the lo<br>s this result, with MS<br>e entire function set.<br>CEC1, MSE-ABC ex<br>n quality and stability,<br>served that the compa<br>79.74) produced subs<br>d at the p < 0.05 leve<br>s superiority is also hig<br>inner position with a m<br>MSE-ABC (21.39) w<br>(5.22), CMA-ES (1.9<br>-ABC was superior in<br>ited with respect to so<br>Pre-|f seven different metaheuristic<br>lts include the best value (Best),<br>on time (Time Mean), Wilcoxon<br>and winner algorithm information<br>ormance differences among the<br>differences between the groups.<br>or pairwise comparisons showed<br>were obtained against at least one<br>confirms that the search dynamics<br>from one another and that the<br>nce.<br>n the mean value (Mean), and the<br>periority by achieving the lowest<br>wed by CMA-ES, which achieved<br>C4, CEC5, and CEC12, while<br>3%), namely CEC10. The ABC,<br>west mean value in any function.<br>E-ABC emerging as the algorithm<br>hibited a distinct superiority over<br>with a mean value of 302.30 and<br>red algorithms (ABC: 38,959.31;<br>tantially higher mean values, and<br>l. The effect size values ranging<br>hly meaningful in practical terms.<br>ean value of 435.30; however, in<br>as considerably higher than those<br>8), and ASRGABC (2.90). This<br>terms of average performance in<br>lution stability.<br>pro|





In the CEC3 and CEC9 functions, MSE-ABC demonstrated perfect stability with standard deviation values of 0.0001 and 0.0000, respectively. While all competing algorithms lagged statistically significantly behind in CEC9 (p = 1.73×10⁻⁶), ASRGABC and SHADE exhibited statistically equivalent performance in CEC3 and were evaluated as a “Tie.” The findings in these two functions reveal that MSE-ABC possesses a stable convergence capability in smooth and low-modal function spaces. In CEC4, however, CMA-ES became the winner; with a mean value of 814.86 and a standard deviation of 4.77, CMA-ES was the only algorithm in this function that simultaneously provided both the lowest mean value and the highest stability. MSE-ABC produced a mean value of 863.76 in this function and lagged significantly behind statistically (p = 1.73×10⁻⁶). This finding suggests that the covariance adaptation mechanism of CMA-ES may provide a structural advantage in certain unimodal or weakly multimodal structures. CEC5 stands out as the function in which MSE-ABC exhibited its weakest performance. In this function, MSE-ABC remained behind all competitors except ASRGABC, with a mean value of 1,642.10 and a standard deviation of 325.98. The winner, CMA-ES, produced both the highestquality and the most stable solution with a mean value of 900.00 and a zero standard deviation. ABC (926.96) also demonstrated competitive performance in this function. The structural characteristics of CEC5—possibly strong local optimum traps or irregular search space geometry—negatively affected the perturbation mechanism of MSE-ABC and revealed that the algorithm is relatively fragile against such function structures. CEC6 emerged as the most difficult function in terms of evaluation; ABC exhibited extremely poor performance in this function with a mean value of 3,585,220.93, while SHADE (157,262.65) and ASRGABC (139,083.56) also produced inconsistent results due to high variance. MSE-ABC, on the other hand, achieved the most stable and highest-quality solution among all competitors with a mean value of 1,851.44 and a standard deviation of only 36.30. The satisfaction of the p < 0.05 significance threshold in all comparisons and the effect size values varying between 0.43 and 0.99 confirm that this superiority is supported by a strong statistical foundation. In the CEC7 and CEC8 functions, MSE-ABC was in the winner position; however, in CEC7, SHADE (p = 0.075) and DOA (p = 0.734), and in CEC8, DOA (p = 0.066), failed to produce statistically significant differences and resulted in a “Tie” decision. This situation demonstrates that even in functions where MSE-ABC outperformed in terms of average performance, certain competing algorithms could still remain within the boundaries of statistical equivalence, empirically supporting that no single algorithm can achieve absolute superiority across the entire function space. A remarkable result was obtained in CEC10: ASRGABC became the winner in this function by producing the lowest mean value of 2,417.85. MSE-ABC exhibited a higher variance in this function with a mean value of 2,466.93 and a standard deviation of 71.28; thus, it is understood that the adaptive search strategy of ASRGABC can improve local search quality in certain multimodal structures. On the other hand, DOA was considered statistically equivalent in this function (p = 0.673). In CEC11, MSE-ABC was the winner with a mean value of 2,783.36; however, the standard deviation of 189.50 in this function corresponds to the highest variability recorded for MSE-ABC across the entire test set. This situation indicates that the algorithm had difficulty achieving consistent convergence in the rugged search landscape of CEC11 and that the performance variability between runs should be evaluated carefully. In CEC12, CMA-ES became the winner; while CMA-ES produced a mean value of 2,937.81, MSE-ABC lagged significantly behind statistically with a mean value of 2,944.13 (p = 0.001). SHADE and DOA were evaluated as statistically equivalent to MSEABC in this function. 

When the average execution times are examined, it is observed that the ABC algorithm resulted in the highest computational cost across all functions (CEC1: 0.419 s, CEC11: 0.731 s, CEC12: 





<!-- Start of picture text -->
w CEC2022 CEC1 - Convergence Curve<br>w Jes,<br>wv<br>s<br>z§ :<br>=<br>ite =<br>5 awe<br>we —<br>oow mew! ' u 1<br><!-- End of picture text -->



<!-- Start of picture text -->
tax CEC2022 CEC4 - Convergence Curve<br>ae<br>—<br>ad a]<br>tose<br>Zz<br>s<br>380<br>eS.<br>vec<br>a " ol = a = on vn wm beat ‘m res<br><!-- End of picture text -->



<!-- Start of picture text -->
ane CEC2022 CECT - Convergence Curve<br>=r<br>ae 7<br>ate 4<br>once<br>eeee<br><!-- End of picture text -->



<!-- Start of picture text -->
ne CEC2022 CEC10 - Convergence Curve<br>ane 7<br>MBENRC<br>sate 4<br>Bras<br>Eee SKC jh<br>\<br>3I0¢ 7<br>a YO<br><!-- End of picture text -->



<!-- Start of picture text -->
CEC2022 CEC2 - Convergence Curve<br>Jes,<br>s<br>zg |<br>h<br>t<br>10<br>ST‘YS — —<br><!-- End of picture text -->



<!-- Start of picture text -->
yn (CEC2022 GECS - Convergence Curve<br>—sRT<br>=,<br>3<br>2s<br>© © io) 200 10 a0 hterationsto 200 0 ato xo' 180<br><!-- End of picture text -->



<!-- Start of picture text -->
w CEC2022 CEC8 - Convergence Curve<br>. — ss<br>ie<br>wt<br>0 ' ' ' '<br><!-- End of picture text -->



<!-- Start of picture text -->
yn CEC2022 CEC11 - Convergence Curve<br>=,<br>2 —— TAOS<br>ats<br>é&<br>as<br>‘ |<br>eration<br><!-- End of picture text -->



<!-- Start of picture text -->
ee CEC2022 CEC3 - Convergence Curve<br>Js,<br>ne<br>re<br>383 a0<br>ste3,oo.- a a aw eke ew<br><!-- End of picture text -->



<!-- Start of picture text -->
w CEC2022 CECE - Convergence Curve<br>=<br>we<br>><br>g<br>8<br>@ ° \—s-s0e<br>.<br>w v w 2c‘ sae' ua IterationWw acu' at' wa Ld<br><!-- End of picture text -->



<!-- Start of picture text -->
orm CEC2022 CECS - Convergence Curve<br>ce on<br>—<br>vost |<br>axe<br>gg, A $_$_$?$?<@_<br><!-- End of picture text -->



<!-- Start of picture text -->
ae CEC2022 CEC 12 - Convergence Curve<br>—<br>7 MBENRC<br>&<br>Ei ssee<br>3006<br>a<br><!-- End of picture text -->



## **Figure 2.** Convergence behavior comparison of the proposed and competing algorithms on CEC-2022 benchmark functions 

Figure 2 presents the convergence behaviors of the proposed MSE-ABC algorithm and the comparative algorithms on the CEC-2022 benchmark functions. The subfigures are arranged sequentially from CEC1 to CEC12 and illustrate how the algorithms improve the objective function values throughout the iterations. The CEC-2022 test suite consists of unimodal, basic, hybrid, and composition functions with different difficulty levels. This structure enables the evaluation of not only convergence speed, but also exploration–exploitation balance, the ability to avoid local minima, robustness in complex and multimodal search spaces, and stability in complex fitness landscapes. Overall, the convergence curves indicate that the MSE-ABC algorithm exhibits a stable and sustainable optimization behavior across different problem categories. CEC1 is a unimodal problem based on the shifted and rotated Zakharov function. For such functions, the primary expectation is that the algorithm should rapidly converge toward the optimum region through strong exploitation capability. The convergence curves demonstrate that MSE-ABC achieves a faster reduction from the early iterations compared to the competing methods and maintains a more stable convergence behavior throughout the optimization process. In particular, ABC and SHADE remain at higher error levels, while SFOA and DOA lose their convergence speed after certain iterations. In contrast, MSE-ABC approaches the optimum region in a controlled and stable manner, demonstrating its strong exploitation capability. CEC2 is based on the rotated Rosenbrock function, which contains strong inter-variable dependency and a narrow valley structure. In such problems, algorithms generally exhibit rapid improvement during the initial iterations but struggle to progress along the optimum valley. The convergence curves show that all algorithms achieve a rapid decline at the beginning; however, MSE-ABC reaches lower error levels in a shorter time. While ASRGABC and CMAES slow down after specific iterations, MSE-ABC maintains a more balanced convergence process. This behavior indicates that the proposed method possesses effective guidance mechanisms in highly correlated search spaces. CEC3 is derived from the expanded Schaffer’s F6 function, which contains intensive multimodality and highly complex fitness landscapes around the optimum region. In such functions, the ability of algorithms to avoid local minima and move toward the global optimum is critically important. The convergence curves indicate that although several algorithms can generate near-optimal solutions, MSE-ABC preserves its early convergence advantage. In particular, SFOA follows a slower improvement process, while some algorithms exhibit plateau behavior. In contrast, MSE-ABC demonstrates a more balanced convergence profile and performs a more effective search within the multimodal landscape. 

CEC1 is a unimodal problem based on the shifted and rotated Zakharov function. For such functions, the primary expectation is that the algorithm should rapidly converge toward the optimum region through strong exploitation capability. The convergence curves demonstrate that MSE-ABC achieves a faster reduction from the early iterations compared to the competing methods and maintains a more stable convergence behavior throughout the optimization process. In particular, ABC and SHADE remain at higher error levels, while SFOA and DOA lose their convergence speed after certain iterations. In contrast, MSE-ABC approaches the optimum region in a controlled and stable manner, demonstrating its strong exploitation capability. 

CEC4 is based on the discontinuous and rotated Rastrigin function, which possesses a highly multimodal and complex search space. Discontinuities and numerous local minima tend to cause premature convergence in optimization algorithms. The convergence curves show that MSE-ABC differentiates itself from the competing algorithms from the early iterations and follows a more balanced improvement process. While ABC and SFOA lose convergence speed in the later iterations, MSE-ABC maintains a more sustainable decline characteristic. This observation demonstrates that the proposed method has a strong exploration mechanism and can preserve diversity in complex search spaces. 

CEC5 is based on the rotated Levy function, which contains irregular fitness surfaces and sharp transition regions, making it a challenging optimization problem. The convergence curves clearly show that the CMAES algorithm reaches the optimum region very rapidly and exhibits 



superior performance throughout the optimization process. Although MSE-ABC does not achieve the best result on this function, it still demonstrates a stable and competitive convergence behavior during the iterations. Compared to the significant fluctuations and irregular improvement behaviors observed in some algorithms, MSE-ABC follows a more controlled optimization trajectory. This result indicates that the proposed method can maintain stable optimization capability even in challenging and irregular fitness landscapes. 

CEC6–CEC8 consist of hybrid benchmark problems. Since hybrid functions are constructed by combining multiple basic functions, they require both exploration and exploitation mechanisms to operate effectively and simultaneously. In particular, the performance differences among algorithms become highly pronounced in CEC6. While methods such as ABC and SHADE exhibit significant fluctuations during the early iterations, MSE-ABC rapidly reaches low error levels and maintains stable convergence behavior. In CEC7 and CEC8, some algorithms tend to converge prematurely or become trapped in local minima. In contrast, MSE-ABC follows a more balanced improvement process and achieves sustainable performance in the complex search spaces of hybrid functions. Especially in CEC8, the low-variance convergence profile clearly demonstrates the robustness of the proposed method. CEC9–CEC12 consist of composition benchmark problems. Composition functions are considered among the most challenging problems in the CEC test suite because they combine multiple sub-functions with different scales and characteristics. In such functions, both global exploration capability and local exploitation performance become critically important. The convergence curves indicate that MSE-ABC exhibits a more balanced and sustainable convergence behavior in most composition functions. In particular, the algorithm reaches solutions very close to the optimum value rapidly in CEC9. Although ASRGABC performs better in CEC10, MSE-ABC maintains its stable convergence behavior throughout the optimization process. Similarly, MSE-ABC demonstrates a significant decline during the early iterations in CEC11, whereas CMAES shows partial superiority in CEC12. Nevertheless, the overall convergence profiles indicate that MSE-ABC can establish a strong exploration– exploitation balance and effectively avoid local minima even in composition problems. Overall, the convergence curves demonstrate that the MSE-ABC algorithm provides fast, stable, and sustainable optimization performance across different categories of CEC-2022 benchmark functions. Particularly for multimodal, hybrid, and composition problems, the proposed method exhibits a more balanced convergence profile and stronger resistance against local optima. The behaviors observed in the convergence curves are largely consistent with the mean, standard deviation, p-value, and effect size results reported in Table 2. The quantitative findings confirm that MSE-ABC achieves statistically significant and reliable performance on the majority of the benchmark functions and stands out among the competing methods in terms of solution accuracy, convergence stability, and overall optimization robustness. **3.2. Parameter Sensitivity Analysis** 

The MSE-ABC algorithm incorporates four new control parameters integrated into the classical ABC framework: momentum coefficient ( _β_ ), exploration probability ( _GP_ ), initial step size ( _step_init_ ), and final step size ( _step_final_ ). In order to systematically evaluate the effect of these parameters on the search behavior of the algorithm and to objectively demonstrate the validity of the default parameter settings, a sensitivity analysis was conducted. Within the scope of the analysis, each parameter was tested using three different values representing low, medium, and high levels, while the remaining parameters were kept fixed at their default settings. The experiments were carried out on four representative functions (F1, F4, F8, and F12) selected from the CEC 2022 benchmark suite by averaging the results of 30 independent runs. These functions represent unimodal, multimodal, hybrid, and composition functions, respectively, 



thereby enabling the observation of parameter effects under different problem geometries. The obtained findings are presented in Table 3. 

**Table 3.** Sensitivity analysis results of the newly introduced parameters of the MSE-ABC algorithm (Mean ± Standard Deviation over 30 independent runs) 

|**Parameter**<br>**Values**<br>**Tested**<br>**F1 (Mean ± Std.)**<br>**F4 (Mean ± Std.)**<br>**F8 (Mean ± Std.)**<br>**F12 (Mean ± Std.)**<br>**Rank**<br>**Stability**|
|---|
|Beta(_β)_<br>[0.1-0.5-0.9]<br>97802.6792 ±<br>26130.4678<br>1085.7697 ±<br>28.9422<br>2607.7361 ±<br>220.8930<br>3231.5436 ±<br>113.8043<br>High|
|<br>GP<br>[0.2-0.5-0.8]<br>101579.5663 ±<br>19864.8050<br>1084.7289 ±<br>25.7757<br>2659.8155 ±<br>412.2091<br>3211.6183 ±<br>111.3841<br>High|
|Step init<br>[0.5-1-2]<br>97449.1577 ±<br>24747.9454<br>1072.9638 ±<br>35.6357<br>2591.1166 ±<br>143.4927<br>3261.2911 ±<br>109.0723<br>High<br>|
|<br> <br> <br>Step final<br>[0.001-0.01-<br>01]<br>104555.2308 ±<br>224735817<br>1079.2316 ±<br>275018<br>2556.6990 ±<br>2145328<br>3188.7536 ±<br>962037<br>High<br>f|
|<br>.<br>.<br>.<br>.<br>.<br> <br>As shown in Table 3, all four parameters preserved the overall ranking performance of the<br>algorithm across the tested value ranges and exhibited high stability. This finding indicates that<br>MSE-ABC does not require excessive parameter tuning in practical applications.<br>**Momentum Coefficient (β)**<br>proo|
|When the β parameter was tested using the values [0.1, 0.5, 0.9], it was observed to produce the<br>lowest result variation on the F1 function with an average value of 97802.68±26130.47. The<br>standard deviation values obtained on F4 and F12 remained at 28.94 and 113.80, respectively,<br>indicating that modifying the momentum coefficient across a wide range does not disrupt search<br>consistency even on multimodal problem surfaces. However, the standard deviation of 220.89<br>observed on F8 suggests that the selection of the β value may relatively influence the<br>convergence speed in complex and multimodal functions; nevertheless, this variation does not<br>statistically produce a significant change in the final solution quality.<br>**Exploration Probability (GP)**<br>Scanning the GP parameter within the range [0.2, 0.5, 0.8] produced the highest standard<br>deviation (412.21) on the F8 function compared with the other parameters. This result reveals<br>that the exploration probability directly governs search diversity in complex and multimodal<br>problems, and therefore parameter selection becomes relatively more important for such<br>problem types. Nevertheless, the obtained mean value (2659.82) remained comparable to those<br>produced by the other parameters and did not lead to a substantial performance degradation.<br>For the F1, F4, and F12 functions, GP yielded consistent results throughout the tested range.<br>**Initial Step Size (step_init)**<br>Journal Pre-|



When the step_init parameter was examined using the values [0.5, 1, 2], it was noteworthy that it produced the lowest mean error value (1072.96) on the F4 function. In addition, the standard deviation obtained on F8 (143.49) remained relatively lower compared with the other parameters, indicating that different initial step size values diversify the algorithm’s early-stage search behavior while maintaining stable solution quality. On the other hand, the highest F12 mean value (3261.29) was obtained under this parameter setting, suggesting that larger initial step sizes may slightly affect the fine exploitation accuracy on hybrid functions without weakening the overall competitiveness of the algorithm. 

## **Final Step Size (step_final)** 



When the step_final parameter was tested within the range [0.001, 0.01, 0.1], the highest mean value (104555.23±22473.58) was observed on the F1 function. This finding suggests that the step size applied toward the final stages of the search process can significantly influence finetuning sensitivity in unimodal problems with broad search spaces. On the other hand, on F8, step_final produced the lowest mean value (2556.70) among all tested parameters together with a relatively moderate standard deviation (214.53), indicating that smaller final step sizes support local refinement on multimodal landscapes. For the F4 and F12 functions, changing the step_final value had a limited effect on performance. 

Overall, the rank stability was determined as “high” for all parameters and all benchmark functions. This result demonstrates that the MSE-ABC algorithm possesses a robust structure against parameter variations and that the proposed default parameter settings provide suitable initial configurations for a broad range of problem classes. In terms of parameter interactions, the combination of GP and step_init was observed to cause relatively higher variation on complex and multimodal functions (F8); therefore, these two parameters may be jointly considered in problems where dimensionality significantly increases. Nevertheless, the obtained results clearly demonstrate that despite the complex structural components of MSEABC, the algorithm does not require extensive parameter optimization from the user, which constitutes an important advantage in terms of practical applicability. **3.3. Performance evaluation of MSE-ABC on low-dimensional and constrained realworld engineering design problems** In this subsection, the performance of the proposed MSE-ABC algorithm in constrained and nonlinear engineering design problems is evaluated in detail. In this context, the Welded Beam, Pressure Vessel, Tension Spring, Speed Reducer, and Three Bar Truss design problems, which are widely used in the literature and represent different levels of difficulty, are addressed. These low-dimensional constrained real-world engineering design problems, with decision variable dimensions ranging from 2 to 7, provide a realistic and challenging testing environment for meta-heuristic algorithms due to their nonlinear constraint structures, narrow feasible solution spaces, and conflicting design objectives. In the experimental evaluations, the objective function and constraints for each engineering design problem were handled as defined in the literature; constraint violations were addressed using a penalty function approach. 



**Table 4.** Mathematical formulations of the low-dimensional and constrained real-world engineering design optimization problems: objective functions, constraint <u>expressions, and decision variable bounds.</u> 

|**Problem**|**Dim.**|**No. of**<br>**Constraints**|**Variable**<br>**Type**|**Objective Function f(x)**|**Constraints  gi(x) ≤ 0**|**Lower**<br>**Bounds**|**Upper**<br>**Bounds**|
|---|---|---|---|---|---|---|---|
|**Tension/**<br>**Compression**<br>**Spring**|3|4|Continuous<br>|min  f(_x_) = (_x_3+ 2)_x_2_x_1<sup>2</sup><br>|g1: 1 −_x_2<sup>3</sup>_x_3/ (71785_x_1<sup>4</sup>) ≤ 0<br>g2: (4_x_2<sup>2</sup>−_x_1_x_2) / [12566(_x_2_x_1<sup>3</sup>−_x_1<sup>4</sup>)] + 1/(5108_x_1<sup>2</sup>)<br>− 1 ≤ 0<br>g3: 1 − 140.45_x_1/ (_x_2<sup>2</sup>_x_3) ≤ 0<br>    <br>f|0.05<br>0.25<br>2.00|2.00<br>1.30<br>15.0|
||||||g4: (_x_1+_x_2) / 1.5−1 ≤ 0<br>|||
|**Pressure Vessel**|4|4|Mixed<br>(discrete +<br>continuous)<br>|min  f(_x_) = 0.6224_x_1_x_3_x_4<br>+ 1.7781_x_2_x_3<sup>2</sup><br>+ 3.1661_x_1<sup>2</sup>_x_4<br>+ 19.84_x_1<sup>2</sup>_x_3<br>|<br>g1: −_x_1+ 0.0193_x_3≤ 0<br>g2: −_x_2+ 0.00954_x_3≤ 0<br>g3: −π_x_3<sup>2</sup>_x_4− (4/3)π_x_3<sup>3</sup>+ 1,296,000 ≤ 0<br>g4:_x_4− 240 ≤ 0<br> <br>pro|0.0625<br>0.0625<br>10<br>10|6.1875<br>6.1875<br>200<br>200|
||||||* _x_1,_x_2: multiples of 0.0625 in.<br>|||
|**Welded Beam**|4|7|Continuous<br>|min  f(_x_) = 1.10471_x_1<sup>2</sup>_x_2<br>+ 0.04811_x_3_x_4(14 +_x_2)<br>l P|<br>g1: τ / τmax− 1 ≤ 0  (τmax= 13,600 psi)<br>g2: σ / σmax− 1 ≤ 0  (σmax= 30,000 psi)<br>g3:_x_1−_x_4≤ 0<br>g4: 0.10471_x_1<sup>2</sup>+ 0.04811_x_3_x_4(14+_x_2) − 5 ≤ 0<br>g5: 0.125 −_x_1≤ 0<br>g6: δ / δmax− 1 ≤ 0  (δmax= 0.25 in)<br>re-|0.125<br>0.1<br>0.1<br>0.125|5.0<br>10.0<br>10.0<br>5.0|
||||||g7: P / Pc −1 ≤ 0<br>|||
|**Speed Reducer**|7|11|Continuous<br>J|min  f(_x_) = 0.7854_x_1_x_2<sup>2</sup>(3.3333_x_3<sup>2</sup><br>+ 14.9334_x_3− 43.0934)<br>− 1.508_x_1(_x_6<sup>2</sup>+_x_7<sup>2</sup>)<br>+ 7.477(_x_6<sup>3</sup>+_x_7<sup>3</sup>)<br>+ 0.7854(_x_4_x_6<sup>2</sup>+_x_5_x_7<sup>2</sup>)<br>ourna|g1: 27 / (_x_1_x_2<sup>2</sup>_x_3) − 1 ≤ 0<br>g2: 397.5 / (_x_1_x_2<sup>2</sup>_x_3<sup>2</sup>) − 1 ≤ 0<br>g3: 1.93_x_4<sup>3</sup>/ (_x_2_x_3_x_6<sup>4</sup>) − 1 ≤ 0<br>g4: 1.93_x_5<sup>3</sup>/ (_x_2_x_3_x_7<sup>4</sup>) − 1 ≤ 0<br>g5: √[(745_x_4/(_x_2_x_3))<sup>2</sup>+1.69×10<sup>7</sup>] / (110_x_6<sup>3</sup>) − 1 ≤ 0<br>g6: √[(745_x_5/(_x_2_x_3))<sup>2</sup>+1.575×10<sup>7</sup>] / (85_x_7<sup>3</sup>) − 1 ≤ 0<br>g7:_x_2_x_3/ 40 − 1 ≤ 0<br>g8: 5_x_2/_x_1− 1 ≤ 0<br>g9:_x_1/ (12_x_2) − 1 ≤ 0<br>g10: (1.5_x_6+ 1.9) /_x_4− 1 ≤ 0<br>   <br>|2.6<br>0.7<br>17<br>7.3<br>7.3<br>2.9<br>5.0|3.6<br>0.8<br>28<br>8.3<br>8.3<br>3.9<br>5.5|
||||||g11: (1.1_x_7+ 1.9) /_x_5−1 ≤ 0|||
|**Three-Bar Truss**|2|3|Continuous|min  f(_x_) = (2√2·_x_1+_x_2)·100|<br>g1: σ1− σallow≤ 0,   σ1= P(√2_x_1+_x_2)/Δ<br>g2: σ2− σallow≤ 0,   σ2= P_x_2/Δ<br>g3: σ3− σallow≤ 0,   σ3= P/(√2_x_2+_x_1)<br>Δ = √2_x_1<sup>2</sup>+ 2_x_1_x_2<br>P=2 kN/cm²,  σallow =2 kN/cm²,  H=100 cm|0.0<br>0.0|1.0<br>1.0|





|Table 4<br>proble<br>dimens<br>design<br>F2 exh<br>constra<br>constra<br>require<br>integer<br>denomi<br>variabl<br>general<br>**Table**<br>constra<br>|summariz<br>ms used in<br>ional deci<br>benchmar<br>ibits a mi<br>ined sprin<br>ined probl<br>ment that t<br>structure,<br>nator term<br>e-type com<br>applicabil<br>**5.**Compar<br>ined real- <br>|es the mathe<br>the evaluat<br>sion variable<br>ks. While all<br>xed-variable<br>g design pro<br>em in this se<br>he variables<br>while in F5<br>approaches<br>binations co<br>ity of MSE-<br>ative statistic<br>world engine<br>|matical char<br>ion of the<br>spaces, and<br>problems ar<br>structure. F<br>blem, where<br>t with seven<br>x1and x2be<br>, numerical<br>zero. The<br>vered by the<br>ABC across<br>al and nume<br>ering design<br>|acteristics o<br>MSE-ABC<br>are treated<br>e primarily b<br>1 represents<br>as F4 consti<br>decision var<br>rounded to<br>stability ass<br>diversity of<br>se problems<br>different eng<br>rical analysi<br>problems<br>|f the five rea<br>algorithm. T<br>as constrai<br>ased on con<br>a low-dim<br>tutes the hig<br>iables and e<br>multiples of<br>urance is a<br>dimensional<br>enables an<br>ineering sea<br>s of algorith<br>|l-world engi<br>he problem<br>ned real-wor<br>tinuous vari<br>ensional and<br>hest-dimens<br>leven constra<br>0.0625 intro<br>pplied for c<br>ity, constrai<br>objective ass<br>rch spaces.<br>ms on low-d<br>of|neering desi<br>s involve lo<br>ld engineeri<br>able structur<br>relatively l<br>ional and m<br>ints. In F2, t<br>duces a mixe<br>ases where t<br>nt density, a<br>essment of t<br>imensional a<br>|gn<br>w-<br>ng<br>es,<br>ess<br>ost<br>he<br>d-<br>he<br>nd<br>he<br>nd|
|---|---|---|---|---|---|---|---|---|
|**Problem**<br>|**Metric**<br>|**ABC**<br>|**ASRGABC**<br>|**SFOA**<br>|**SHADE**<br>|**DOA**<br>|**CMAES**<br>|**MSEABC**|
||Mean<br>|2.128841886<br>|2.088804267<br>|1.724851693<br>|2.479539264<br>|3.267792342<br>|3.493470605<br>|1.724851692|
|**Welded Beam**<br>|Std<br>|0.119740228<br>|0.243652395<br>|1.90416E-09<br>|0.624034154<br>|0.536619345<br>|0.8293018<br>|4.42618E-10|
||Time (ms)<br>|457.4073533<br>|110.5746767<br>|77.35318<br>|361.10713<br>-|65.67319<br>|83.50483<br>|132.3596667|
||Mean<br>|6415.060437<br>|6178.147542<br>|6050.924698<br>|6524.826135<br>|7304.235989<br>|6581.290098<br>|6050.130281|
|**Pressure Vessel**<br>|Std<br>|142.7374817<br>|102.6277434<br>|5.842804064<br>r|482.1271233<br>|488.5095468<br>|359.0909064<br>|1.491599373|
||Time (ms)<br>|429.14904<br>|90.85969<br>|59.85414333<br>|346.21576<br>|49.97348<br>|68.92014<br>|103.0158967|
||Mean<br>|0.013172385<br>|0.012748207<br>|0.012665423<br>|0.012953167<br>|11169.3122<br>|0.012909404<br>|0.012665245|
|**Tension Spring**<br>|Std<br>|0.000129307<br>|4.95656E-05<br>|4.2006E-08<br>|0.000490204<br>|60176.12744<br>|0.000444459<br>|3.32111E-07|
||Time (ms)<br>|446.0646967<br>|95.84039333<br>|64.37741333<br>|351.26494<br>|55.08528667<br>|71.20808333<br>|109.36076|
||Mean<br>|2815.196553<br>|2815.196553<br>|2815.196553<br>|2815.239966<br>|2817.42906<br>|2825.802829<br>|2815.196553|
|**Speed Reducer**<br>|Std<br>|2.36551E-07<br>|5.53739E-13<br>|5.06667E-13<br>|0.235022739<br>|1.657428121<br>|3.739386594<br>|4.62521E-13|
||Time (ms)<br>|518.6896433<br>|131.9205367<br>|93.2969<br>|392.77666<br>|91.74237667<br>|115.3511267<br>|175.9155933|
||Mean<br>|263.8939609<br>|263.8980407<br>|263.8914911<br>|263.8915335<br>|264.2909934<br>|263.8914911<br>|263.8914911|
|**Three Bar Truss**<br>|Std<br>|0.00244714<br>|0.006461935<br>|1.1563E-13<br>|0.000218425<br>|0.336261428<br>|1.29279E-13<br>|1.1563E-13|
||Time (ms)<br>|404.9748067<br>|81.68811333<br>|51.06701333<br>|349.8128733<br>|41.66127<br>|57.61287333<br>|82.78277667|
|**Total Rank**<br>||20.50<br>|18.50<br>|10.50<br>|25.00<br>|33<br>|25.00<br>|7.50|
|**Average Rank**<br>||4.10<br>|3.70<br>|2.10<br>|5.00<br>|6.6<br>|5.00<br>|1.50|
|**Final Order**||4|3|2|5|7|5|1|
|<br>Table 5<br>enginee<br>comput<br>or near<br>and Pr<br>indicati<br>exhibit<br>extrem|presents<br>ring desi<br>ational tim<br>-best objec<br>essure Ves<br>ng strong<br>s the weak<br>ely large o|<br>a comprehe<br>gn problems<br>e. The resul<br>tive values i<br>sel problem<br>exploitation<br>est solution q<br>bjective valu|<br>nsive compa<br>in terms o<br>ts show that<br>n most benc<br>s, both meth<br>capability an<br>uality, partic<br>es.|<br>rison of sev<br>f mean obj<br>MSEABC a<br>hmark proble<br>ods converg<br>d effective c<br>ularly in the|<br>en optimiza<br>ective valu<br>nd SFOA c<br>ms. In parti<br>e to nearly<br>onvergence<br>Tension Spr|<br>tion algorith<br>e, standard<br>onsistently a<br>cular, for the<br>identical opt<br>behavior. In<br>ing problem,|<br>ms across fi<br>deviation, a<br>chieve the b<br>Welded Be<br>imal solutio<br>contrast, DO<br>where it yie|<br>ve<br>nd<br>est<br>am<br>ns,<br>A<br>lds|



The standard deviation results highlight the robustness of the algorithms. MSEABC and SFOA demonstrate exceptionally low variance across all benchmark problems, with near-zero 



standard deviation values observed especially in the Three-Bar Truss problem. This indicates highly stable and consistent convergence behavior. ASRGABC shows a clear improvement over the classical ABC algorithm in both mean performance and stability. SHADE and CMAES exhibit moderate performance, achieving competitive results in some cases but failing to consistently match the best-performing algorithms. DOA shows the least stable behavior, particularly in the Tension Spring problem, where it presents extremely high variance, indicating poor robustness. 

The Friedman ranking analysis provides a statistical summary of overall algorithm performance across all test problems. According to the results, MSEABC achieves the best overall performance with an average rank of 1.50, followed by SFOA (2.10) and ASRGABC (3.70). ABC obtains a moderate rank of 4.10, while SHADE and CMAES achieve identical average ranks of 5.00. DOA ranks last with an average rank of 6.60, confirming its inferior performance compared to the other methods. These results indicate that, according to the Friedman test, MSEABC is the most statistically consistent and competitive algorithm among all compared methods. In terms of computational time, noticeable differences are observed among the algorithms. Although DOA achieves relatively low execution time in some cases, its poor solution quality limits its overall effectiveness. SFOA and ASRGABC generally exhibit lower computational cost, whereas ABC and SHADE tend to require higher execution times in several problems. CMAES shows moderate-to-high computational cost with limited performance gain. Overall, when considering solution quality, stability, and statistical ranking together, MSEABC emerges as the most balanced and superior algorithm among the compared approaches. **3.4. Performance evaluation of MSE-ABC for deep learning hyperparameter optimization** In this subsection, the performance of the proposed MSE-ABC algorithm in the deep learningbased hyperparameter tuning problem is evaluated on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60.000 color images of 32 × 32 pixels belonging to 10 different classes, containing 50.000 training and 10,000 test examples. Due to its visual complexity and high similarity between classes, CIFAR-10 provides a challenging testing environment for deep learning models. In the experiments, a ResNet-50 architecture adapted to the CIFAR-10 dataset was employed, as detailed in Tables 7 and 8. The model was trained from scratch without using any pre-trained ImageNet weights, with all parameters randomly initialized. Since no prior knowledge is transferred to the model, the classification performance observed across all runs depends entirely on the hyperparameter configuration found by each optimization algorithm. This allows a direct and unbiased comparison of the competing optimization algorithms. Training ResNet50 from scratch on CIFAR-10 is a methodologically standard approach in the hyperparameter optimization literature. The model was trained using stochastic gradient descent with crossentropy loss, and the learning rate, batch size, number of epochs, and momentum coefficient were treated as the four decision variables to be optimized. 

All algorithms were run under the same network architecture, training protocol, and experimental parameters to ensure fair and reproducible evaluation. Experiments were conducted with 30 independent runs; population size and maximum iteration count were kept equal for all methods. The algorithms included in the comparison were implemented using the default parameter settings recommended in the literature. 



The lower and upper bounds of the hyperparameters optimized for the CNN model were determined by considering the ranges commonly used in the deep learning and hyperparameter optimization literature. The relevant bounds are presented in Table 6. 

**Table 6.** Lower and upper bounds of CNN hyperparameters for the CIFAR-10 dataset. 

|**Hyperp**|**arameter**|**Descri**|**ption**|**Lower Bound**|**Upper Bound**|
|---|---|---|---|---|---|
|Learnin|g rate|Step si|ze for weight|updates<br>0.0001|0.1|
|Batch si|ze|Numbe|r of samples|per batch<br>16|256|
|Epochs||Numbe|r of training|iterations<br>20|200|
|Momen<br>|tum<br>|Mome<br>|ntumcoeffici<br>|entforSGD<br>0.2<br>|0.99<br>|
|Table 7<br>hyperpa<br>of this<br>model a<br>This ar<br>convolu<br>are prog<br>employ<br>sequent<br>to obtai<br>output v<br>**Table 7**<br>|and Table<br>rameter tun<br>architecture<br>dapted to th<br>chitecture b<br>tional layer<br>ressively re<br>ed to enha<br>ially to 256<br>n a 2048-di<br>ia a fully c<br>**.**ResNet-5<br>|8 present<br>ing and t<br>. Table 7<br>e CIFAR-<br>egins wi<br>and conti<br>duced wh<br>nce repre<br>, 512, 10<br>mensiona<br>onnected<br>0 for CIF<br>|in detail<br>he bottlene<br>illustrates<br>10 dataset,<br>th the pro<br>nues throu<br>ile channe<br>sentational<br>24, and 204<br>l feature ve<br>layer.<br>AR-10 arch<br>|the architecture of the CNN mod<br>ck block that constitutes the fund<br>the overall network structure of<br>showing how layers are organized<br>cessing of 32×32×3 input image<br>gh four-stage residual blocks, whe<br>l depth is increased. In each stage,<br>capacity, and the number of c<br>8. In the final stage, global avera<br>ctor, which is then transformed in<br>itecture<br>Pre-pro|el optimized through<br>amental building unit<br>the ResNet-50–based<br>from input to output.<br>s through the initial<br>re spatial dimensions<br>bottleneck blocks are<br>hannels is increased<br>ge pooling is applied<br>to a 10-class softmax<br>of|
|**Stage**<br>|**Layer**<br>**Type**<br>|**Output**<br>**Size**<br>|**Kernel /**<br>**Stride**<br>|**Channel Flow (in →**<br>**mid→ out)**<br>**Activation**<br>|**Blocks**<br>|
|Input<br>|Image<br>|32 × 32<br>× 3<br>|-<br>a|-<br>-<br>|-<br>|
|Conv1<br>|Conv +<br>BN<br>|32 × 32<br>× 64<br>|3×3 / s=1<br>|3 → 64<br>ReLU<br>|1<br>|
|Stage<br>1<br>|Bottleneck<br>|32 × 32<br>× 256<br>u|s=1<br>|64 → 64 → 256<br>BN +<br>ReLU<br>|3<br>|
|Stage<br>2<br>|Bottleneck<br>|16 × 16<br>× 512<br>|s=2 (first<br>block)<br>|256 → 128 → 512<br>BN +<br>ReLU<br>|4<br>|
|Stage<br>3<br>|Bottleneck<br>J|8 × 8 ×<br>1024<br>|s=2 (first<br>block)<br>|512 → 256 → 1024<br>BN +<br>ReLU<br>|6<br>|
|Stage<br>4|Bottleneck|4 × 4 ×<br>2048|s=2 (first<br>block)|1024 → 512 → 2048<br>BN +<br>ReLU|3|
|GAP|Global<br>Avg Pool|1 × 1 ×<br>2048|-|-<br>-|-|
|FC|Fully<br>Connected|10|-|2048 → 10<br>Softmax|1|



Table 8, on the other hand, provides a detailed description of the bottleneck block, which serves as the fundamental building component of this architecture. Accordingly, the block consists of 1×1, 3×3, and 1×1 convolutions in sequence; in the first layer, the number of channels is reduced to decrease computational cost, in the second layer spatial features are extracted, and in the final layer the channel dimension is expanded again to preserve representational capacity. In cases of dimensional mismatch, a projection shortcut is used to align the input and output 



features, and an element-wise summation is performed via the residual connection followed by activation. 

**<u>Table 8.</u>** <u>Bottleneck Block</u> 

|**Step**<br>**Operation**<br>**Kernel**<br>**Channel Mapping**<br>**Activation**|
|---|
|1<br>Conv + BN<br>1×1<br>C→C/4<br>ReLU|
|2<br>Conv + BN<br>3×3<br>C/4→C/4<br>ReLU|
|3<br>Conv + BN<br>1×1<br>C/4→C<br>None|
|Shortcut<br>Identity / Projection<br>1×1 (only if mismatch)<br>Match dimensions<br>-|
|Merge<br>Element-wise Sum<br>-<br>-<br>ReLU<br>|
|The quantitative results of the experiments conducted on the CIFAR-10 dataset are presented<br>in Table 9. The table provides the average classification accuracy, standard deviation, best and<br>worst accuracy values, and p-values according to MSE-ABC for each algorithm. The baseline<br>CNN model without hyperparameter tuning exhibited both low performance and significant<br>instability, with an average accuracy of 85.4% and a high standard deviation of 2.1. This result<br>demonstrates that hyperparameter choices directly determine the overall behavior and<br>generalization ability of deep neural networks.<br>roof|
|**Table 9.**Comparative results of CNN hyperparameter optimization on the CIFAR-10 dataset.<br>**Algorithm**<br>**Mean Acc. (%)**<br>**Std. (%)**<br>**Best (%)**<br>**Worst (%)**<br>**p-value (vs Proposed)**<br>-|
|Baseline CNN(No HPO)<br>85.40<br>2.1<br>88.2<br>82.0<br>7.1×10⁻⁵<br>|
|Random Search<br>87.32<br>1.75<br>89.3<br>85.2<br>6.5×10⁻⁴<br>|
|ABC<br>88.63<br>1.3<br>90.3<br>86.1<br>1.2×10⁻⁴<br>|
|SHADE<br>88.87<br>1.2<br>90.5<br>86.9<br>9.5×10⁻⁵<br>|
|ASRGABC<br>89.04<br>1.1<br>90.8<br>87.2<br>2.6×10⁻⁵<br>|
|DOA<br>89.14<br>1.0<br>90.9<br>87.3<br>1.1×10⁻⁶<br>|
|CMAES<br>90.12<br>1.1<br>91.8<br>89.1<br>3.8×10⁻³<br>|
|Bayesian Opt.<br>90.67<br>1.2<br>91.9<br>89.7<br>4.1×10⁻³<br>|
|SFOA<br>91.33<br>0.9<br>92.8<br>89.7<br>2.1×10⁻²<br>|
|MSE-ABC(Proposed)<br>93.16<br>0.8<br>94.1<br>91.9<br>—<br>|
|HPO processes performed with meta-heuristic optimization algorithms have achieved an<br>average accuracy increase of 3–4 percentage points across all models and provided a significant<br>improvement in model stability by reducing the standard deviation to the range of 0.8–1.3. This<br>improvement can be attributed to more effective exploration of the hyperparameter space and<br>a more balanced learning process.<br>Although Random Search achieved a mean accuracy of 87.32% with a standard deviation of<br>1.75, it remained the weakest among the HPO methods due to its unguided stochastic nature,<br>which limits its ability to exploit promising regions of the hyperparameter space. Similarly,<br>Bayesian Optimization achieved a competitive mean accuracy of 90.67%; however, its standard<br>deviation of 1.2 and p-value of 4.1×10⁻³ indicate that MSE-ABC still provides a statistically<br>significant improvement over this surrogate-based method. These results suggest that neither<br>random exploration nor probabilistic surrogate modeling alone is sufficient to match the<br>performance of a well-designed hybrid metaheuristic approach. Although the search space<br>comprises only four hyperparameters, the highly nonlinear and non-convex fitness landscape<br>arising from the interaction among learning rate, batch size, epoch count, and momentum makes<br>this a nontrivial optimization problem. Consequently, population-based adaptive search<br>mechanisms can provide a more effective exploration–exploitation balance than Random<br>Jou|



HPO processes performed with meta-heuristic optimization algorithms have achieved an average accuracy increase of 3–4 percentage points across all models and provided a significant improvement in model stability by reducing the standard deviation to the range of 0.8–1.3. This improvement can be attributed to more effective exploration of the hyperparameter space and a more balanced learning process. 



Search, which relies on unguided stochastic sampling, and Bayesian Optimization, which depends on probabilistic surrogate modeling strategies. 

The proposed hybrid method outperformed all other approaches in terms of both performance and stability, achieving an average accuracy of 93.16%, a standard deviation of 0.8, and a best accuracy value of 94.1%. The decrease in standard deviation from 2.1 to 0.8 demonstrates that the proposed method not only finds the best result but also provides a reliable optimization strategy that produces consistent and repeatable results in the hyperparameter space. This finding strongly supports the idea that while meta-heuristic algorithms are effective on their own, performance can be further enhanced by hybridizing complementary features. 

The superiority of the proposed method is also supported by statistical significance tests. The results of the paired t-test performed with 10-fold cross-validation showed a statistically significant difference compared to all competing algorithms, with p-values ranging from 2.1×10⁻² (SFOA) to 1.1×10⁻⁶ (DOA). It is worth noting that DOA yielded the smallest p-value (1.1×10⁻⁶) among all competitors despite having a relatively small mean accuracy gap, which can be attributed to its low standard deviation of 1.0 reducing within-group variance in the paired test. The comparison with the closest competitor SFOA yielded p = 2.1×10⁻², confirming that the improvement of MSE-ABC is statistically significant even against the second-best performing algorithm. 

Journal Pre-proof 

|airplane<br>#P||.936<br>MeL|fae<br> 0.09%|8<br> 0.08%|5<br> 0.05%|5<br> 0.05%|5<br> 0.05%|5<br> 0.05%|5<br> 0.05%|12<br>10<br> 0.12% 0.10%|
|---|---|---|---|---|---|---|---|---|---|
|TL|8<br>0)09%|Eeynm<br> EM ELA|5<br> 0.05%|4<br> 0.04%|3<br> 0.03%|4<br> 0.04%|3<br> 0.03%|5<br> 0.02%|10<br>16<br> 0.10% 0.16%|
|bird|Mew<br>0.12%|3<br> 0.03%|ym<br> fwPLA|is<br> 0.18%|15<br> 0.15%|8<br> 0.08%|7<br> 0.07%|10<br> 0.10%|5<br>z<br> 0.05% 0.02%|
|cat|WE<br>0.05%|3<br> 0.03%|20<br> 0.20%|Mem<br> EMA|15<br> 0.15%|25<br> 0.25%|12<br> 0.12%|8<br> 0.08%|4<br>3<br> 0.04% 0.03%|
|a<br>&<br>deer<br>he<br>|Jae<br>0.06%|2<br> 0.02%|15<br> 0.15%|12<br> 0.12%|[Ekim<br> [MePLA|10<br> 0.10%|6<br> 0.06%|12<br> 0.12%|4<br>3<br> 0.04% 0.03%|
|o<br>doa<br>3<br>0<br>|Mae<br>.04%|2<br> 0.02%|10<br> 0.10%|28<br> 0.28%|10<br> 0.10%|MRM<br> EWE]|6<br> 0.06%|15<br> 0.15%|5<br>5<br> 0.05% 0.05%|
|jal<br>_<br>ll|3<br> ).03%|2<br> 0.02%|8<br> 0.08%|10<br> 0.10%|6<br> 0.06%|ee<br> 0.05%|950<br> EespLy|fae<br> 0.06%|5<br>5<br> 0.05% 0.05%|
|horse|ae<br>0.05%|2<br> 0.02%|8<br> 0.08%|6<br> 0.06%|10<br> 0.10%|12<br> 0.12%|4<br> 0.04%|Peyum<br> EMBL|5<br>8<br> 0.05% 0.08%|
|eT<br>0|10<br>.10%|8<br> 0.08%|3<br> 0.03%|4<br> 0.04%|3<br> 0.03%|3<br> 0.03%|3<br> 0.03%|2<br> 0.02%|yam<br>14<br> Feeeve 0.14%|
|truck|We<br>0.08%|15<br> 0.15%|y)<br> 0.02%|3<br> 0.03%|3<br> 0.03%|5<br> 0.05%|3<br> 0.03%|6<br> 0.06%|iy<br>925<br> 0.12% WAL|
|S|&<br>s<br>S<br>S|s||Cc<br>|6|er|s<br>S|e|x<br>SS|



True Class 



the dataset, supporting the notion that MSE-ABC HPO makes the decision surface more meaningful. In this context, hyperparameter optimization not only improves accuracy but also limits the model's misclassification tendencies in a more controlled and consistent manner. The concentration of error patterns in challenging classes such as cat–dog does not overshadow the model's overall success; rather, it shows that errors remain limited within a logical class similarity framework. These results demonstrate that MSE-ABC with HPO contributes to the CNN learning intra-class variations better and making the error distribution more regular. 

## **Conclusion** 

In this study, MSE-ABC, a synergistic hybrid of the ABC and SFOA algorithms, has been proposed, and its performance has been comprehensively evaluated in the context of CEC 2022 benchmark functions, real-world constrained engineering design problems, and deep learning hyperparameter optimization. Unlike conventional hybridization approaches based on simple operator exchange or sequential execution, MSE-ABC establishes a complementary and unified search strategy that explicitly exploits the robust local exploitation capability of ABC and the strong global exploration behavior of SFOA within a single adaptive framework. The hybrid design of MSE-ABC goes beyond a mechanical combination by incorporating a momentumbased learning mechanism that accumulates historical search information and provides directed guidance to exploration operators. In the early stages of the search process, the wide-angle exploratory movements inspired by SFOA enable an effective and diversified scan of the solution space. As the search progresses, the accumulated momentum information progressively biases the search toward ABC’s exploitation-oriented dynamics, thereby accelerating convergence while maintaining stability. This dynamic transition allows MSEABC to continuously regulate the exploration–exploitation balance rather than relying on fixed or iteration-dependent control rules. The conducted experiments and non-parametric statistical analyses demonstrate that MSE-ABC consistently produces best or near-best solutions, particularly in highly constrained and multimodal search spaces. Despite its hybrid structure and the inclusion of additional search operators, the algorithm’s average computational times remain on the same order as those of competing methods, which constitutes a noteworthy outcome from a computational efficiency perspective. This observation indicates that the additional mechanisms integrated into MSEABC have been designed in a computationally efficient manner and do not introduce a significant increase in the overall algorithmic burden. Consequently, the achieved performance gains are obtained at a practically acceptable and largely negligible additional computational cost. The Friedman rankings and pairwise Wilcoxon tests confirm that the observed performance differences are not attributable to random variation and that MSE-ABC exhibits a systematic advantage across a wide range of problems. At the same time, the results clearly indicate that no single algorithm achieves absolute dominance across all problem instances. In particular, the strong competitiveness of certain methods, such as SFOA, on specific problem classes reaffirms the problem-dependent nature of performance in metaheuristic optimization. Moreover, despite having been introduced nearly two decades ago, the ABC algorithm remains competitive with contemporary methods on many problems, highlighting the enduring potential of well-designed foundational heuristics. 

Nevertheless, this study is subject to several inevitable limitations. First, compared to classical ABC, MSE-ABC incorporates a larger number of structural components, which may increase the complexity of parameter interactions, particularly when transferring the algorithm to substantially different problem classes. Second, although the selected engineering design problems are widely accepted benchmarks in the literature, they do not fully represent complex, 



multimodal, dynamic, or time-varying constrained environments. Finally, in the deep learning experiments, the optimization process was conducted primarily with a performance-oriented focus, while aspects such as training stability, energy consumption, and hardware awareness were not explicitly incorporated into the optimization framework. 

Rather than weakening the contributions of this work, these limitations delineate concrete and meaningful directions for further investigation. In particular, instead of further increasing algorithmic complexity, adaptive strategies that enable the selective and online activation of MSE-ABC components may enhance its generalizability. Furthermore, multi-objective formulations that jointly consider computational cost and solution quality could extend the applicability of MSE-ABC to modern engineering and learning-based systems. Finally, a theoretical analysis of the interactions among algorithmic components and their convergence behavior remains an open and challenging research problem of considerable importance. In summary, MSE-ABC is positioned as a practical and applicable metaheuristic approach that places balance at its core, achieving a meaningful trade-off among solution quality, statistical reliability, and computational efficiency. The reported results strongly suggest that computationally conscious hybrid designs can deliver repeatable and substantive performance improvements across both classical engineering problems and contemporary learning-based applications. **Author Contribution:** The entire manuscript was conceived, written, and finalized solely by the author. **Conflict of interest:** The author declares that there is no conflict of interest associated with this work. **References** [1] Akbulut, H. (2026). A modified starfish optimization algorithm (M-SFOA) for global optimization problems and its application to heart disease risk prediction. Expert Systems With Applications, 307, 131088. https://doi.org/10.1016/j.eswa.2026.131088 [2] Deng, L., Qiu, Y., Di, Y., & Zhang, L. (2025). A knowledge-driven memetic algorithm for distributed green flexible job shop scheduling considering the endurance of machines. Applied Soft Computing, 170, 112697. https://doi.org/10.1016/j.asoc.2025.112697 [3] Tang, H., Fang, B., Liu, R., Li, Y., & Guo, S. (2022). A hybrid teaching and learning-based optimization algorithm for distributed sand casting job-shop scheduling problem. Applied Soft Computing, 120, 108694. https://doi.org/10.1016/j.asoc.2022.108694 [4] Lin, J., Shi, C., Jin, J., Li, S., & Chen, D. (2025). A hybrid trajectory optimization solution applied to UAVs based on point cloud information and bio-inspired evolutionary algorithm. Applied Soft Computing, 189, 114470. https://doi.org/10.1016/j.asoc.2025.114470 [5] Zhang, X., Chen, B., Xiao, J., & Yang, J. (2025). Advanced metaheuristic optimization with enhanced dung beetle algorithm for automated crack detection in civil infrastructure. Applied Soft Computing, 190, 114548. https://doi.org/10.1016/j.asoc.2025.114548 

[6] Özdemir, R., Taşyürek, M., & Aslantaş, V. (2026). An improved marine predators algorithm for shipment status time estimation and regression problems. International Journal of Machine Learning and Cybernetics, 17(2). https://doi.org/10.1007/s13042-025-02967-5 

[7] Karaboga, D., & Basturk, B. (2007). On the performance of artificial bee colony (ABC) algorithm. Applied Soft Computing, 8(1), 687–697. https://doi.org/10.1016/j.asoc.2007.05.007 

[8] Özdemir, R., Taşyürek, M., & Aslantaş, V. (2025). EL-NRF: Enhancing ensemble learning for regression with a noise reduction framework. Expert Systems With Applications, 286, 128074. https://doi.org/10.1016/j.eswa.2025.128074 

[9] Aslantas, V., & Kurban, R. (2010). Fusion of multi-focus images using differential evolution algorithm. Expert Systems With Applications, 37(12), 8861–8870. https://doi.org/10.1016/j.eswa.2010.06.011 



[10]Durmus, A., Yildirim, Z., Kurban, R., & Karakose, E. (2024). An optimal concentric circular antenna array design using atomic orbital search for communication systems. Frequenz, 78(9–10), 543–558. https://doi.org/10.1515/freq-2023-0432 

[11]Kiran, M. S., & Hakli, H. (2020). A tree–seed algorithm based on intelligent search mechanisms for continuous optimization. Applied Soft Computing, 98, 106938. https://doi.org/10.1016/j.asoc.2020.106938 

[12]Durmus, A., & Kurban, R. (2021). Optimum design of linear and circular antenna arrays using equilibrium optimization algorithm. International Journal of Microwave and Wireless Technologies, 13(9), 986–997. https://doi.org/10.1017/s1759078720001774 

[13]Kaya, E., Kaya, C. B., Bendeş, E., Atasever, S., Öztürk, B., & Yazlık, B. (2023). Training of Feed-Forward neural networks by using optimization algorithms based on Swarm-Intelligent for maximum power point tracking. Biomimetics, 8(5), 402. https://doi.org/10.3390/biomimetics8050402 [14]Karaboga, D., & Kaya, E. (2016). An adaptive and hybrid artificial bee colony algorithm (aABC) for ANFIS training. Applied Soft Computing, 49, 423–436. https://doi.org/10.1016/j.asoc.2016.07.039 [15]Kiran, M. S. (2015). The continuous artificial bee colony algorithm for binary optimization. Applied Soft Computing, 33, 15–23. https://doi.org/10.1016/j.asoc.2015.04.007 [16]Kıran, M. S., & Fındık, O. (2014). A directed artificial bee colony algorithm. Applied Soft Computing, 26, 454–462. https://doi.org/10.1016/j.asoc.2014.10.020 [17]Aslan, M., Gunduz, M., & Kiran, M. S. (2019). JayaX: Jaya algorithm with xor operator for binary optimization. Applied Soft Computing, 82, 105576. https://doi.org/10.1016/j.asoc.2019.105576 [18]Kıran, M. S., & Gündüz, M. (2013). A recombination-based hybridization of particle swarm optimization and artificial bee colony algorithm for continuous optimization problems. Applied Soft Computing, 13(4), 2188– 2203. https://doi.org/10.1016/j.asoc.2012.12.007 [19]Akay, B., & Karaboga, D. (2010). A modified Artificial Bee Colony algorithm for real-parameter optimization. Information Sciences, 192, 120–142. https://doi.org/10.1016/j.ins.2010.07.015 [20]Wang, R., Pan, J., Chu, S., Lin, B., & Zhong, N. (2026). A Multi-Strategy Population-Free Particle Swarm Optimization Algorithm under Gamma Distribution in a Fixed Sample Domain and Its Comprehensive Analysis. Applied Soft Computing, 114578. https://doi.org/10.1016/j.asoc.2026.114578 [21]Sharma, R., Matharu, J. S., & Parmar, K. S. (2025). A survey on Particle Swarm Optimization: Evolution, adaptations and practical implementations. Applied Soft Computing, 186, 114016. https://doi.org/10.1016/j.asoc.2025.114016 [22]Zhang, S., Yuan, P., Zhang, C., & Pan, M. (2025). Sparse identification of partial differential equations via collaborative ant colony optimization and physics-informed neural networks. Expert Systems With Applications, 299, 130253. https://doi.org/10.1016/j.eswa.2025.130253 [23]Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. Journal of Global Optimization, 39(3), 459–471. https://doi.org/10.1007/s10898-007-9149-x [24]Xiang, W. L., Meng, X. L., Li, Y. Z., He, R. C., & An, M. Q. (2018). An improved artificial bee colony algorithm based on the gravity model. Information Sciences, 429, 49–71. https://doi.org/10.1016/j.ins.2017.11.007 [25]Yildirim, M. Y., & Akay, R. (2025). An efficient grid-based path planning approach using improved artificial bee colony algorithm. Knowledge-Based Systems, 318, 113528. https://doi.org/10.1016/j.knosys.2025.113528 [26]Zeng, T., Wang, W., Wang, H., Cui, Z., Wang, F., Wang, Y., & Zhao, J. (2022). Artificial bee colony based on adaptive search strategy and random grouping mechanism. Expert Systems With Applications, 192, 116332. https://doi.org/10.1016/j.eswa.2021.116332 

[27]Mao, J.-Y., Pan, Q.-K., Miao, Z.-H., & Gao, L. (2021). An effective multi-start iterated greedy algorithm for the distributed permutation flowshop scheduling. Expert Syst. Appl., 169, 114495. 

[28]Zhuang, M.Z., Zhang, W., Tang, H.T., Li, X.Y., & Wang, K.P. (2024). A multi-objective genetic algorithm based on two-stage reinforcement learning. Expert Syst. Appl., 258, 125189. 

[29]Saini, G., & Jadon, S. S. (2025). An improved artificial bee colony algorithm based on spider monkey optimization global search for complex benchmarks and engineering applications. Physica Scripta, 100(7), 075229. 

[30]Zhou, X., Tan, G., Wang, H., Ma, Y., & Wu, S. (2024). Artificial bee colony algorithm based on multineighbor guidance. Expert Systems With Applications, 259, 125283. https://doi.org/10.1016/j.eswa.2024.125283 



[31]Zhou, X., Lu, J., Huang, J., Zhong, M., & Wang, M. (2021). Enhancing artificial bee colony algorithm with multi-elite guidance. Information Sciences, 543, 242–258. https://doi.org/10.1016/j.ins.2020.07.037 

[32]Zeng, T., Wang, W., Wang, H., Cui, Z., Wang, F., Wang, Y., & Zhao, J. (2021). Artificial bee colony based on adaptive search strategy and random grouping mechanism. Expert Systems With Applications, 192, 116332. https://doi.org/10.1016/j.eswa.2021.116332 

[33]Li, Y., Huang, W., Wu, R., & Guo, K. (2020). An improved artificial bee colony algorithm for solving multiobjective low-carbon flexible job shop scheduling problem. Applied Soft Computing, 95, 106544. https://doi.org/10.1016/j.asoc.2020.106544 

[34]Xiang, W., Meng, X., Li, Y., He, R., & An, M. (2017). An improved artificial bee colony algorithm based on the gravity model. Information Sciences, 429, 49–71. https://doi.org/10.1016/j.ins.2017.11.007 

[35]Cui, Y., Hu, W., & Rahmani, A. (2022). A reinforcement learning based artificial bee colony algorithm with application in robot path planning. Expert Systems With Applications, 203, 117389. https://doi.org/10.1016/j.eswa.2022.117389 [36]Wu, R., Luo, E., Li, X., Tang, H., & Li, Y. (2025). Hybrid artificial bee colony algorithm with Q-learning for distributed heterogeneous flexible job shop scheduling problem considering machine preventive maintenance. Swarm and Evolutionary Computation, 98, 102134. https://doi.org/10.1016/j.swevo.2025.102134 [37]Liu, M., Yuan, Y., Xu, A., Deng, T., & Jian, L. (2024). A learning-based artificial bee colony algorithm for operation optimization in gas pipelines. Information Sciences, 690, 121593. https://doi.org/10.1016/j.ins.2024.121593 [38]Dinesh, A., & Rangaraj, J. (2025). An energy-efficient routing protocol for wireless body area networks using hybrid artificial bee colony optimization and chicken swarm optimization algorithm. Journal of Engineering and Applied Science, 72(1). https://doi.org/10.1186/s44147-024-00533-4 [39]Angelov, A., & Lazarova, M. (2025). Hybrid Artificial Bee Colony Algorithm for test case generation and optimization. Algorithms, 18(10), 668. https://doi.org/10.3390/a18100668 [40]Zhu, G., & Kwong, S. (2010). Gbest-guided artificial bee colony algorithm for numerical function optimization. Applied Mathematics and Computation, 217(7), 3166–3173. https://doi.org/10.1016/j.amc.2010.08.049 [41]Li, Z., Li, Z., Wang, W., Yan, Y., Li, Z., & Li, Z. (2015). PS–ABC: A hybrid algorithm based on particle swarm and artificial bee colony for high-dimensional optimization problems. Expert Systems With Applications, 42(22), 8881–8895. https://doi.org/10.1016/j.eswa.2015.07.043 [42]Jadon, S. S., Tiwari, R., Sharma, H., & Bansal, J. C. (2017). Hybrid Artificial Bee Colony algorithm with Differential Evolution. Applied Soft Computing, 58, 11–24. https://doi.org/10.1016/j.asoc.2017.04.018 [43]Liang, Z., Hu, K., Zhu, Q., & Zhu, Z. (2017). An enhanced artificial bee colony algorithm with adaptive differential operators. Applied Soft Computing, 58, 480–494. https://doi.org/10.1016/j.asoc.2017.05.005 [44]Chen, X., Tianfield, H., & Li, K. (2019). Self-adaptive differential artificial bee colony algorithm for global optimization problems. Swarm and Evolutionary Computation, 45, 70–91. https://doi.org/10.1016/j.swevo.2019.01.003 [45]Wang, C., Shang, P., & Shen, P. (2022). An improved artificial bee colony algorithm based on Bayesian estimation. Complex & Intelligent Systems, 8(6), 4971–4991. https://doi.org/10.1007/s40747-022-00746-1 [46]Aslan, S., & Arslan, S. (2022). A modified artificial bee colony algorithm for classification optimisation. International Journal of Bio-Inspired Computation, 20(1), 11. https://doi.org/10.1504/ijbic.2022.126280 [47]Ustun, D., Toktas, A., Erkan, U., & Akdagli, A. (2022). Modified artificial bee colony algorithm with differential evolution to enhance precision and convergence performance. Expert Systems With Applications, 198, 116930. https://doi.org/10.1016/j.eswa.2022.116930 [48]Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2024). Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers. Neural Computing and Applications, 37(5), 3641–3683. https://doi.org/10.1007/s00521-024-10694-1 

[49]Ghosh, A., Das, S., Das, A. K., Senkerik, R., Viktorin, A., Zelinka, I., & Masegosa, A. D. (2022). Using spatial neighborhoods for parameter adaptation: An improved success history based differential evolution. Swarm and Evolutionary Computation, 71, 101057. https://doi.org/10.1016/j.swevo.2022.101057 

[50]Lang, Y., & Gao, Y. (2025). Dream Optimization Algorithm (DOA): A novel metaheuristic optimization algorithm inspired by human dreams and its applications to real-world engineering problems. Computer Methods in Applied Mechanics and Engineering, 436, 117718. https://doi.org/10.1016/j.cma.2024.117718 

[51]Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). Reducing the Time Complexity of the Derandomized Evolution Strategy with Covariance Matrix Adaptation (CMA-ES). Evolutionary Computation, 11(1), 1–18. https://doi.org/10.1162/106365603321828970 

[52]Zeng, T., Wang, W., Wang, H., Cui, Z., Wang, F., Wang, Y., & Zhao, J. (2021b). Artificial bee colony based on adaptive search strategy and random grouping mechanism. Expert Systems With Applications, 192, 116332. https://doi.org/10.1016/j.eswa.2021.116332 





<!-- Start of picture text -->
10° CEC2022 CEC1 - Convergence Curve<br>—— ABC<br>—— ASRGABC<br>—— SFOA<br>10° ———— SHADEDOA<br>—— CMAES<br>——— MSEABC<br>107<br>cy<br>© 406<br>8 10<br>Da<br>2<br>o<br>@ 10°<br>10°<br>102<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->





<!-- Start of picture text -->
CEC2022 CEC2 - Convergence Curve<br>—— ABC<br>——— ASRGABC<br>—— SFOA<br>—— SHADE<br>—— DOA<br>104 —— CMAES<br>——— MSEABC<br>~|<br>2S|oO<br>2oo> |<br>o<br>o<br>oO<br>&<br>iL<br>103<br>NN—S=—, = =<br>) 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->



<!-- Start of picture text -->
760 CEC2022 CEC3 - Convergence Curve<br>—— ABC<br>—— ASRGABC<br>—— SFOA<br>740 —— SHADE<br>——DOA<br>—— CMAES<br>————= MSEABC<br>720<br>700<br>oO<br>2<br>i)<br>> |<br>% 680<br>oO<br>c<br>=<br>Ww<br>660<br>640 +}<br>— ———————&—&I=&&=<br>600 Su<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->





<!-- Start of picture text -->
1200 CEC2022 CEC4 - Convergence Curve<br>—— ABC<br>—— ASRGABC<br>—— SFOA<br>1150 —— SHADE<br>—— DOA<br>—— CMAES<br>——— MSEABC<br>1100<br>|<br>1050 |<br>oO<br>=<br>©<br>><br>#% 1000 :<br>@\<br>=<br>i950 = .<br>900 Q as . L| | |<br>_—_———__—— =<br>850 ——_<br>800<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->



<!-- Start of picture text -->
95 X10" CEC2022 CECS5 - Convergence Curve<br>—— ABC<br>—— ASRGABC<br>—— SFOA<br>—— SHADE<br>—— DOA<br>2 ————— CMAESMSEABC<br>@ 1.5<br>2<br>o<br>><br>)<br>o<br>5)<br>c<br>= |<br>wu 4<br>d\<br>a _——_ cS |<br>———————<br>0<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->





<!-- Start of picture text -->
107 CEC2022 CEC6 - Convergence Curve<br>108<br>C<br>©<br>oO<br>”<br>fo?)<br>2 105<br>wo —— ABC<br>8 — ASRGABC<br>B=: — SFOA<br>iL —— SHADE<br>=—— DOA<br>m= CMAES<br>——= MSEABC<br>104<br>103<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->



<!-- Start of picture text -->
5700 CEC2022 CEC7 - Convergence Curve<br>ee ABC<br>——— ASRGABC<br>—— SFOA<br>2600 AE<br>== CMAES<br>=== MSEABC<br>2500<br>8 soo<br>2 2400<br>n<br>tt) }<br>3FS |<br>© 2300<br>LL \<br>2200 \<br>100 Na a<br>2000<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->





<!-- Start of picture text -->
10° CEC2022 CEC8 - Convergence Curve<br>=————= ABC<br>————= ASRGABC.<br>—— SFOA<br>== SHADE<br>—— DOA<br>—— CMAES<br>== MSEABC<br>10°<br>iC}<br>o<br>oO<br>oo<br>SD<br>xe)<br>oo<br>oO<br>c<br>=<br>Le<br>104<br>10°ToL<br>) 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->



<!-- Start of picture text -->
6500 CEC2022 CEC9 - Convergence Curve<br>————= ABC<br>—— ASRGABC<br>6000 —— SFOA<br>—— SHADE<br>——= DOA<br>=——= CMAES<br>5500 == MSEABC<br>5000<br>o<br>=<br>i) 4500<br>><br>“”<br>“”<br>oO<br>£ 4000<br>Wwe<br>3500<br>25003000 |we<br>2000<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->





<!-- Start of picture text -->
9000 CEC2022 CEC10 - Convergence Curve<br>—— ABC<br>————= ASRGABC<br>—— SFOA<br>8000 ee<br>—— CMAES<br>== MSEABC<br>7000 |<br>2 |<br>= 6000<br>”<br>ry)<br>3= |<br>= 5000<br>Le \<br>4000 \<br>3000<br>. oS<br>Se<br>2000<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->



<!-- Start of picture text -->
95 X10 CEC2022 CEC11 - Convergence Curve<br>——— ABC<br>=———— ASRGABC<br>—— SFOA<br>men SHADE<br>—— DOA<br>2 ——= CMAESMSEABC<br>@ 1.5<br>2<br>&<br>><br>7]<br>no<br>fe)<br>=<br>a<br>uw 4<br>0<br>0 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->





<!-- Start of picture text -->
5000 CEC2022 CEC12 - Convergence Curve<br>—— ABC<br>——— ASRGABC<br>——SFOA<br>———= SHADE<br>——DOA<br>4500 ———— CMAESMSEABC<br>@ 4000<br>a}<br>©<br>><br>“”<br>”<br>E<br>=<br>c ol<br>Le 3500<br>MS<br>3000 Ri<br>2500 0) 100 200 300 400 500 600 700 800 900 1000<br>Iteration<br><!-- End of picture text -->

