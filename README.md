# GATE_DA
# Data Science and Artificial Inteligence
1. Probability and Statistics
2. Linear Algebra
3. Calculus and Optimization
4. Programming , Data Structure and Algorithm
5. Database Management and Warehousing
6. Machine Learning
7. AI
   
## Paper Structure
- Total Marks: 100
- Duration: 3 hours
#### Sections: 
- General Aptitude (15 marks)
- Technical (85 marks)
##### Question Types:
- Multiple Choice Questions (1 mark)
- Multiple Choice Questions (2 marks)
- Numerical Answer Type (NAT)

### Section 1. Mathematics & Statistics (~20-25 marks)
#### Linear Algebra
  Key Focus Areas:

1. Matrix Operations
   Important Topics:
   - Rank computation
   - Matrix inverse
   - Matrix decomposition
   - Eigenvalue problems
   
   Common Questions:
   - Find eigenvalues/eigenvectors
   - Solve system of equations
   - Matrix properties verification

2. Vector Spaces
   Key Concepts:
   - Basis and dimension
   - Linear independence
   - Orthogonality
   - Projection matrices
## Example Problems & Formulas 
   - 1. Matrix Properties:
   - Rank(AB) ≤ min(rank(A), rank(B))
   - tr(AB) = tr(BA)
   - det(AB) = det(A)det(B)

2. Eigenvalue Problems:
   Steps:
   1. Find characteristic equation: |A - λI| = 0
   2. Solve for λ
   3. Find eigenvectors: (A - λI)v = 0

3. SVD Decomposition:
   A = UΣV^T
   where:
   - U: left singular vectors
   - Σ: singular values
   - V: right singular vectors
     
#### 2. Probability & Statistics  (7-8 marks)
Key Areas:

###### 1. Probability Fundamentals
   - Conditional Probability
   - Bayes' Theorem
   - Independence
   
   Formulas:
   P(A|B) = P(A∩B)/P(B)
   P(A∩B) = P(A|B)P(B)
   Bayes: P(A|B) = P(B|A)P(A)/P(B)

###### 2. Distributions
   Types:
   a) Discrete:
      - Binomial: n trials, p success
      - Poisson: rare events
      
   b) Continuous:
      - Normal: μ, σ²
      - Exponential: rate λ
      
   Properties:
   Normal: 
   - 68-95-99.7 rule
   - Z-score = (x-μ)/σ

#### 3. Statistical Testing (5-7 marks)
Important Concepts:

1. Hypothesis Testing
   Steps:
   1. State H₀ and H₁
   2. Choose significance level α
   3. Calculate test statistic
   4. Compare with critical value
   
   Common Tests:
   - Z-test
   - t-test
   - Chi-square test
   
2. Confidence Intervals
   Formula:
   CI = x̄ ± z(α/2)(σ/√n)
   where:
   - x̄: sample mean
   - z: z-score
   - σ: standard deviation
   - n: sample size
### Important Formulas to Remember
   1. Linear Algebra:
   - |AB| = |A||B|
   - (A^-1)^T = (A^T)^-1
   - Rank + Nullity = n

2. Probability:
   - E(aX + b) = aE(X) + b
   - Var(aX + b) = a²Var(X)
   - P(A∪B) = P(A) + P(B) - P(A∩B)

3. Statistics:
   - Z = (X - μ)/σ
   - t = (X̄ - μ)/(s/√n)
   - χ² = Σ((O-E)²/E)
     
#### Calculus & Optimization
  • Derivatives & Gradients
  • Multivariate Calculus
  • Optimization Techniques

 ### Section 2: Machine Learning (~20-25 marks) 
#### 1. Supervised Learning (12-15 marks)
  • Linear Regression
  • Logistic Regression
  • Decision Trees
  • Random Forests
  • SVM
  • KNN

  Key Areas:

1. Linear Models
   a) Linear Regression
      Formulas:
      - MSE = (1/n)Σ(y - ŷ)²
      - R² = 1 - (SSres/SStot)
      - β = (X^T X)^(-1)X^T y
      
      Important Concepts:
      - Assumptions
      - Regularization (L1, L2)
      - Feature scaling
   
   b) Logistic Regression
      Formulas:
      - P(y=x) = 1/(1 + e^(-wx))
      - Loss = -Σ(y log(ŷ) + (1-y)log(1-ŷ))
      
      Key Points:
      - Decision boundary
      - Threshold selection
      - Multi-class extension

2. Tree-Based Methods
   a) Decision Trees
      Metrics:
      - Entropy = -Σp(i)log₂p(i)
      - Gini = 1 - Σp(i)²
      - Information Gain
      
   b) Random Forests
      Concepts:
      - Bagging
      - Feature importance
      - OOB error

3. Support Vector Machines
   Formulas:
   - Margin = 2/||w||
   - Kernel trick: K(x,y) = φ(x)·φ(y)
   
   Types:
   - Linear SVM
   - Non-linear SVM (RBF, Polynomial)

#### 2. Unsupervised Learning (5-7 marks)
  • K-means Clustering
  • Hierarchical Clustering
  • PCA
  • Dimensionality Reduction

  Important Topics:

1. Clustering
   a) K-means
      Algorithm Steps:
      1. Initialize centroids
      2. Assign points
      3. Update centroids
      4. Repeat until convergence
      
      Metrics:
      - Inertia
      - Silhouette score
   
   b) Hierarchical Clustering
      Types:
      - Agglomerative
      - Divisive
      
      Linkage Methods:
      - Single
      - Complete
      - Average
      - Ward's

2. Dimensionality Reduction
   a) PCA
      Steps:
      1. Standardize data
      2. Compute covariance matrix
      3. Find eigenvectors
      4. Project data
      
      Formulas:
      - Explained variance ratio
      - Component selection

#### 3.  Model Evaluation (3-5 marks)
  • Cross-validation
  • Metrics (Precision, Recall, F1)
  • ROC Curves
  • Bias-Variance Tradeoff

  Key Concepts:

1. Performance Metrics
   Classification:
   - Accuracy = (TP + TN)/(TP + TN + FP + FN)
   - Precision = TP/(TP + FP)
   - Recall = TP/(TP + FN)
   - F1 = 2(P×R)/(P+R)
   
   Regression:
   - MSE, RMSE, MAE
   - R-squared
   - Adjusted R-squared

2. Validation Techniques
   Methods:
   - K-fold cross-validation
   - Stratified K-fold
   - Leave-one-out
   
   Formula:
   CV score = (1/K)Σ(validation_score)

### 3. Deep Learning (~15-20 marks)
#### Neural Networks Fundamentals (6-8 marks)
  • Feedforward Networks
  • Backpropagation
  • Activation Functions
  • Loss Functions

Key Concepts:
1. Network Architecture
   Components:
   - Input layer
   - Hidden layers
   - Output layer
   
   Formulas:
   - Forward propagation: z = wx + b
   - Activation: a = f(z)
   - Loss computation: L(ŷ,y)

2. Activation Functions
   Types & Derivatives:
   
   ReLU:
   f(x) = max(0,x)
   f'(x) = {1 if x>0, 0 otherwise}
   
   Sigmoid:
   f(x) = 1/(1 + e^(-x))
   f'(x) = f(x)(1-f(x))
   
   tanh:
   f(x) = (e^x - e^(-x))/(e^x + e^(-x))
   f'(x) = 1 - tanh²(x)

3. Backpropagation
   Steps:
   1. Forward pass
   2. Compute loss
   3. Calculate gradients
   4. Update weights
   
   Formula:
   w = w - α∂L/∂w

#### 2. CNN Architecture (5-6 marks)
  • Convolution Operations
  • Pooling Layers
  • CNN Architectures

  Important Components:

1. Convolution Layer
   Calculations:
   - Output size = ((W-F+2P)/S) + 1
   Where:
   W = input size
   F = filter size
   P = padding
   S = stride
   
   Types:
   - Valid padding
   - Same padding
   - Full padding

2. Pooling Layer
   Operations:
   - Max pooling
   - Average pooling
   - Global pooling
   
   Formula:
   Output size = ⌊(n+2p-f)/s⌋ + 1

3. Popular Architectures
   - LeNet
   - AlexNet
   - VGG
   - ResNet
   Key features and innovations

#### 3. RNN & LSTM (4-5 marks)
  • Sequential Data Processing
  • Vanishing Gradient
  • LSTM Architecture
Core Concepts:

1. RNN Basics
   Formulas:
   - h_t = tanh(W_hh·h_(t-1) + W_xh·x_t + b_h)
   - y_t = W_hy·h_t + b_y
   
   Issues:
   - Vanishing gradient
   - Exploding gradient
   - Long-term dependencies

2. LSTM Architecture
   Gates:
   - Forget gate: f_t = σ(W_f·[h_(t-1),x_t] + b_f)
   - Input gate: i_t = σ(W_i·[h_(t-1),x_t] + b_i)
   - Output gate: o_t = σ(W_o·[h_(t-1),x_t] + b_o)
   
   Cell state:
   - C_t = f_t * C_(t-1) + i_t * C̃_t

#### Advanced Topics (2-3 marks)
  • Transfer Learning
  • GANs
  • Transformers
Key Areas:

1. Transfer Learning
   Approaches:
   - Feature extraction
   - Fine-tuning
   - Domain adaptation

2. GANs
   Components:
   - Generator
   - Discriminator
   
   Loss functions:
   - Generator loss
   - Discriminator loss

3. Transformers
   Key mechanisms:
   - Self-attention
   - Multi-head attention
   - Position encoding

### Important Formulas & Concepts
1. Neural Networks:
   - Weight updates: w = w - α∇L
   - Gradient calculation
   - Learning rate scheduling

2. CNN:
   - Receptive field calculation
   - Feature map size
   - Number of parameters

3. RNN/LSTM:
   - Sequence processing
   - Gradient flow
   - Gate mechanisms

4. Optimization:
   - Adam: m_t = β₁m_(t-1) + (1-β₁)g_t
   - RMSprop: v_t = βv_(t-1) + (1-β)g_t²
   - Learning rate decay
### Section 4. Programming & Data Structures (~10-15 marks)
#### Python Programming (5-6 marks)
  Key Concepts:

1. Python Basics
   Important Topics:
   - Data types & operations
   - Control structures
   - List comprehensions
   - Lambda functions
   
   Common Questions:
   ```python
   # List comprehension
   [x for x in range(10) if x%2==0]
   
   # Lambda function
   lambda x: x*x
   
   # Generator expression
   (x**2 for x in range(10))
   ```

2. OOP Concepts
   Core Areas:
   - Classes & Objects
   - Inheritance
   - Polymorphism
   - Encapsulation
   
   Example:
   ```python
   class DataProcessor:
       def __init__(self, data):
           self.data = data
           
       def process(self):
           return [x*2 for x in self.data]
   ```

3. NumPy & Pandas Operations
   Key Functions:
   ```python
   # NumPy
   np.array()
   np.reshape()
   np.concatenate()
   
   # Pandas
   pd.DataFrame()
   df.groupby()
   df.merge()
   ```

#### 2. Data Structures (3-4 marks)
  • Arrays
  • Lists
  • Trees
  • Graphs
  Important Concepts:

1. Linear Structures
   Arrays:
   - Time complexity
   - Space complexity
   - Operations
   
   Operations    | Array | Linked List
   --------------|-------|-------------
   Access        | O(1)  | O(n)
   Insert Begin  | O(n)  | O(1)
   Insert End    | O(1)* | O(1)
   Delete Begin  | O(n)  | O(1)
   Delete End    | O(1)* | O(1)
   Search        | O(n)  | O(n)

2. Tree Structures
   Binary Trees:
   - Traversals
   - Height calculation
   - Balancing
   
   BST Operations:
   - Insertion: O(log n)
   - Deletion: O(log n)
   - Search: O(log n)

3. Graph Representations
   Types:
   - Adjacency Matrix
   - Adjacency List
   
   Space Complexity:
   - Matrix: O(V²)
   - List: O(V+E)

#### 3. Algorithms (3-4 marks)
  • Sorting
  • Searching
  • Graph Algorithms
Key Areas:

1. Sorting Algorithms
   Comparison:
   Algorithm    | Time (Avg) | Time (Worst) | Space
   ------------|------------|--------------|-------
   Quick Sort  | O(nlogn)   | O(n²)       | O(logn)
   Merge Sort  | O(nlogn)   | O(nlogn)    | O(n)
   Heap Sort   | O(nlogn)   | O(nlogn)    | O(1)

2. Searching
   Methods:
   - Linear Search: O(n)
   - Binary Search: O(log n)
   - Hash Table: O(1) average

3. Graph Algorithms
   Common Algorithms:
   - BFS: O(V+E)
   - DFS: O(V+E)
   - Dijkstra: O(V log V + E)
### Section 5. Data Management & Processing (~10-15 marks)
#### 1. Databases (4-5 marks)
  • SQL
  • NoSQL
  • Data Modeling
Important Topics:

1. SQL
   Key Concepts:
   - Joins
   - Aggregations
   - Subqueries
   
   Example Queries:
   ```sql
   -- Complex Join
   SELECT e.name, d.dept_name
   FROM employees e
   LEFT JOIN departments d
   ON e.dept_id = d.id
   WHERE e.salary > (
       SELECT AVG(salary)
       FROM employees
   );
   ```

2. NoSQL
   Types:
   - Document (MongoDB)
   - Key-Value (Redis)
   - Column-family (Cassandra)
   - Graph (Neo4j)

3. Data Modeling
   Concepts:
   - ER diagrams
   - Normalization
   - Indexing
#### Big Data Processing (3-4 marks)
  • Hadoop
  • Spark
  • MapReduce
Key Concepts:

1. Hadoop Ecosystem
   Components:
   - HDFS
   - MapReduce
   - YARN
   
   MapReduce Paradigm:
   Map: (k1,v1) → list(k2,v2)
   Reduce: (k2,list(v2)) → list(v3)

2. Spark
   Features:
   - RDD operations
   - DataFrame API
   - SparkSQL
   
   Example:
   ```python
   # Spark DataFrame
   df.groupBy("column")\
     .agg({"value": "mean"})\
     .show()
   ```
#### Data Processing (3-4 marks)
  • ETL
  • Data Cleaning
  • Feature Engineering
  Important Areas:

1. ETL Process
   Steps:
   - Extract: Data collection
   - Transform: Data cleaning
   - Load: Data storage
   
   Best Practices:
   - Data validation
   - Error handling
   - Performance optimization

2. Feature Engineering
   Techniques:
   - Scaling
   - Encoding
   - Feature selection
   
   Methods:
   - StandardScaler
   - OneHotEncoder
   - PCA

### Important Concepts to Remember

     1. Python:
   - List vs Tuple vs Set
   - Dictionary operations
   - Generator functions
   - Decorators

2. Data Structures:
   - Time complexity
   - Space complexity
   - Trade-offs

3. Databases:
   - ACID properties
   - Index types
   - Query optimization

## Syllabus

## Probability and Statistics:
Counting (permutation and combinations), probability axioms, Sample space, events, independent events, mutually exclusive events, marginal, conditional and joint probability, Bayes Theorem, conditional expectation and variance, mean, median, mode and standard deviation, correlation, and covariance, random variables, discrete random variables and probability mass functions, uniform, Bernoulli, binomial distribution, Continuous random variables and probability distribution function, uniform, exponential, Poisson, normal, standard normal, t-distribution, chi-squared distributions, cumulative distribution function, Conditional PDF, Central limit theorem, confidence interval, z-test, t-test, chi-squared test. 

## Linear Algebra: 
Vector space, subspaces, linear dependence and independence of vectors, matrices, projection matrix, orthogonal matrix, idempotent matrix, partition matrix and their properties, quadratic forms, systems of linear equations and solutions; Gaussian elimination, eigenvalues and eigenvectors, determinant, rank, nullity, projections, LU decomposition, singular value decomposition.

## Calculus and Optimization: 
Functions of a single variable, limit, continuity and differentiability, Taylor series, maxima and minima, optimization involving a single variable. 

## Programming, Data Structures and Algorithms: 
Programming in Python, basic data structures: stacks, queues, linked lists, trees, hash tables; Search algorithms: linear search and binary search, basic sorting algorithms: selection sort, bubble sort and insertion sort; divide and conquer: mergesort, quicksort; introduction to graph theory; basic graph algorithms: traversals and shortest path.

## Database Management and Warehousing:
ER-model, relational model: relational algebra, tuple calculus, SQL, integrity constraints, normal form, file organisation, indexing, data types, data transformation such as normalisation, discretization, sampling, compression; data warehouse modelling: schema for multidimensional data models, concept hierarchies, measures: categorization and computations. 

## Machine Learning:
(i) Supervised Learning: regression and classification problems, simple linear regression, multiple linear regression, ridge regression, logistic regression, k-nearest neighbour, naive Bayes classifier, linear discriminant analysis, support vector machine, decision trees, bias,variance trade-off, cross-validation methods such as leave-one-out (LOO) cross-validation, k-folds cross-validation, multi-layer perceptron, feed-forward neural network; (ii) Unsupervised Learning: clustering algorithms, k-means/k-medoid, hierarchical clustering, top-down, bottom-up: single linkage, multiple-linkage, dimensionality reduction, principal component analysis.

## AI: 
Search: informed, uninformed, adversarial; logic, propositional, predicate; reasoning under uncertainty topics — conditional independence representation, exact inference through variable elimination, and approximate inference through sampling.

### Important Resources
1.Books
- Pattern Recognition and Machine Learning (Bishop)
- Deep Learning (Goodfellow)
- Introduction to Statistical Learning
- Python for Data Analysis

2. Online Resources
- Stanford CS229 (Machine Learning)
- Stanford CS231n (CNN)
- Fast.ai Courses
- Kaggle Competitions

3. Python Libraries:
  • NumPy
  • Pandas
  • Scikit-learn
  • TensorFlow/PyTorch
