class NCF_Data(object):
	"""
	 Construct Dataset for NCF
	"""
	def __init__(self, params, ratings):
		self.ratings = ratings
		self.num_ng = params['num_ng']
		self.num_ng_test = params['num_ng_test']
		self.batch_size = params['batch_size']

		self.preprocess_ratings = self._reindex(self.ratings)

		self.user_pool = set(self.ratings['user_id'].unique())
		self.item_pool = set(self.ratings['item_id'].unique())

		self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
		self.negatives = self._negative_sampling(self.preprocess_ratings)
		random.seed(params['seed'])
