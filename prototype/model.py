import torch, torch.nn as nn
import torch.nn.functional as F

class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim=768, hidden=512, att_dim=512, dropout=0.2):
        super().__init__()
        # Step 1: tile projection
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Step 2: gated attention parameters
        self.V = nn.Linear(hidden, att_dim, bias=True)
        self.U = nn.Linear(hidden, att_dim, bias=True)
        self.w = nn.Linear(att_dim, 1, bias=False)
        # Step 4: slide classifier
        self.cls = nn.Linear(hidden, 1)

    def forward(self, F_bag, mask=None, topk=None, temperature=1.0):
        """
        F_bag: (n, d) float32  -- tile features for ONE slide
        mask : (n,) bool or None  -- if you padded, mark valid tiles as True
        topk : int or None -- if set, keep only top-k attention scores
        """
        # 1. Tile Projection
        H = self.embed(F_bag)                    # (n, h)
        # 2. Gated Attention
        A_tanh = torch.tanh(self.V(H))           # (n, a)
        A_sig  = torch.sigmoid(self.U(H))        # (n, a)
        s = self.w(A_tanh * A_sig).squeeze(1)    # (n,)

        if mask is not None:                     # mask out pads
            s = s.masked_fill(~mask, float('-inf'))

        if topk is not None and topk < s.shape[0]:
            # keep only top-k, set others to -inf so softmax ~0
            kth = torch.topk(s, k=topk, sorted=False).values.min()
            s = torch.where(s >= kth, s, torch.full_like(s, float('-inf')))

        # numerical stability: subtract max before softmax
        a = F.softmax((s - torch.nan_to_num(s.max())) / temperature, dim=0)  # (n,)

        # 3. Bag Aggregation
        M = torch.sum(a.unsqueeze(1) * H, dim=0)  # (h,)

        # 4. Classification
        z = self.cls(M).squeeze(0)                # ()
        p = torch.sigmoid(z)
        return z, p, a, M
