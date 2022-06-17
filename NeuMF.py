class NeuMF(nn.Module):
    def __init__(self, params, num_users, num_items):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = params['factor_num']
        self.factor_num_mlp =  int(params['layers'][0]/2)
        self.layers = params['layers']
        self.dropout = params['dropout']
        self.use_bilinear1 = params['bilinear1']
        self.use_bilinear2 = params['bilinear2']

        self.bilinear1 = nn.Bilinear(self.factor_num_mlp, self.factor_num_mlp, self.factor_num_mlp*2)
        self.bilinear2 = nn.Bilinear(params['layers'][-1], self.factor_num_mf, 128)

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(params['layers'][:-1], params['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=params['layers'][-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        if not self.use_bilinear1:
            mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        else:
            mlp_vector = self.bilinear1(user_embedding_mlp, item_embedding_mlp)

        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)
        # print(mlp_vector.shape, user_embedding_mf.shape, item_embedding_mf.shape, mf_vector.shape)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        if not self.use_bilinear2:
            vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        else:
            vector = self.bilinear2(mlp_vector, mf_vector)
            vector = torch.cat([mlp_vector, mf_vector, vector], dim=-1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()