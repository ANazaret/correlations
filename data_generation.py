class DataGenerator():
    def __init__(self, num_cells, num_cell_types, num_genes, K, S):
        self.num_cells = num_cells
        self. num_cell_types = num_cell_types
        self.num_genes = num_genes
        self.K = K
        self.S = S
        self.reset()
    def reset(self):
        self.gen_knn_graph()
        self.gen_cell_types()
        self.gen_perturbation()
        self.gen_cell_caracteristics()
        self.gen_perturbed_caracteristics()
    def gen_knn_graph(self):
        points = np.random.uniform(size=(self.num_cells,2))
        dists = np.sum(points**2, axis = 1)[:,None] - 2* points @ points.T +np.sum(points**2, axis = 1)[None,:]
        self.graph = np.array([np.argsort(dists[i])[1:self.K+1] for i in range(self.num_cells)])
        
    def gen_cell_types(self):
        self.cell_types =  np.random.choice(self.num_cell_types, size=self.num_cells)

    def gen_perturbation(self):
        shape = (self.num_cell_types, self.num_cell_types, self.num_genes)
        self.perturbation = 0.25*(2*np.random.binomial(1, 0.5, size = shape)-1)*np.random.beta(0.3,0.3,size = shape) 
    
    def gen_cell_caracteristics(self):
        self.initial_cell_carac = 0.75*np.random.beta(1,1, size=(self.num_cell_types, self.num_genes))
        
    def gen_perturbed_caracteristics(self):
        raw_cell_type_graph = [np.unique(self.cell_types[self.graph][i], return_counts = True) for i in range(self.num_cells)]
        effective_perturbation = np.array([np.sum([count*self.perturbation[self.cell_types[i]][val] for (val, count) in zip(*line)], axis = 0) for i,line in enumerate(raw_cell_type_graph)])/K
        self.effective_gene_expression = self.initial_cell_carac[self.cell_types]+effective_perturbation
        self.effective_gene_expression *= (self.effective_gene_expression >= 0)
    
    def generate_gene_expression(self):

        return np.array([[np.random.poisson(self.S*v) for v in line] for line in self.effective_gene_expression])
