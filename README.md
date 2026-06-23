# Strategies to Avoid Poor Local Minima in Nonlinear Optimization
This seminar paper was written in Spring 2026 for the course "Seminar on Methodological Foundations of Operations Research" at the chair for Continuous Optimization of the Karlsruhe Institute of Technology (KIT) Institute of Operations Research. It examines the tunneling algorithm as presented by Levy and Montalvo (1985). 
The seminar paper includes a deeper dive into the mathematical foundations of the algorithm and a Python implementation of the algorithm. 
Insights from the theoretical examination of the algorithm were applied in improving the algorithm, most importantly by changing the movable pole logic and implementing simulated annealing.

The file structure is as follows:
```text
├── Submissions/       # Final documents and completed seminar paper
├── latex/             # Overleaf project source files
├── python/            # Implementation code
│   ├── accompanying_code.ipynb  # Main analysis
│   ├── tunneling_algorithm.py   # Core implementation with and without improvements
│   ├── algo_tester.py           # Algorithm test method
│   ├── functions.py             # Collection of test functions
│   └── requirements.txt
├── .gitignore
└── README.md
```

The code analysis of the algorithms is saved in a Jupyter Notebook and can be viewed by clicking on `python/accompanying_code.ipynb`.
