


        # item_emb1 = dyemb
        # input_emb1 = item_emb1 + cas_emb
        # input_emb1 = self.LayerNorm(input_emb1)
        # input_emb1 = self.dropout(input_emb1)
        #
        # position_ids = torch.arange(input.size(1), dtype=torch.long, device=input.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input)
        # position_embedding = self.position_embedding(position_ids.cuda())
        # # item_emb2 = self.item_embedding(input.cuda())
        # item_emb2 = all_emb
        # input_emb2 = item_emb2 + position_embedding
        # input_emb2 = self.LayerNorm(input_emb2)
        # input_emb2 = self.dropout(input_emb2)
        # input_emb = self.fus(input_emb1, input_emb2)
        # extended_attention_mask = self.get_attention_mask(input)
        # trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)
        #
        # # output = self.fus2(dyemb, trm_output)
        # pred = self.pred(trm_output)
        # mask = get_previous_user_mask(input.cpu(), self.n_node)
        #
        # return (pred + mask).view(-1, pred.size(-1)).cuda()
