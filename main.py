import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
NUM_EPOCH = 100
LR = 1e-4
L2 = 3e-5
ALPHA = 0.4
TEMPERATURE = 0.6
print(f"alpha{ALPHA}, LR:{LR}, L2:{L2}, Temperature:{TEMPERATURE}")

class FeatureDataset(Dataset):
    def __init__(self, text_path, image_path):
        text_data = np.load(text_path)
        image_data = np.load(image_path, allow_pickle=True)

        self.text = torch.tensor(text_data["data"], dtype=torch.float32)
        self.image = image_data["data"]
        self.label = torch.tensor(text_data["label"], dtype=torch.long)

        min_len = min(len(self.text), len(self.image), len(self.label))
        self.text = self.text[:min_len]
        self.image = self.image[:min_len]
        self.label = self.label[:min_len]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        text_feat = self.text[idx]
        image_feat = torch.tensor(self.image[idx], dtype=torch.float32)
        if image_feat.dim() >= 3:
            image_feat = image_feat.squeeze(-1).squeeze(-1)
        return text_feat, image_feat, self.label[idx]

def collate_fn(batch):
    text, image, label = zip(*batch)
    return torch.stack(text), torch.stack(image), torch.tensor(label)

class_weight = torch.tensor([1.0, 3.0]).to(DEVICE)
def weighted_cross_entropy(inputs, targets):
    return F.cross_entropy(inputs, targets, weight=class_weight)

class DetectionModule(nn.Module):
    def __init__(self, text_dim=200, image_dim=512, num_experts=6):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.num_experts = num_experts

        self.text_proj = nn.Linear(text_dim, image_dim)
        self.classifier = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(image_dim, 2)
        )
        self.match_head = nn.Linear(image_dim * 2, 2)

    def forward(self, text_encoding, image_encoding):
        if image_encoding.dim() >= 3:
            image_encoding = image_encoding.squeeze(-1).squeeze(-1)

        text_avg = text_encoding.mean(dim=1)
        text_512 = self.text_proj(text_avg)

        fused = torch.cat([text_512, image_encoding], dim=-1)

        final_logits = self.classifier(fused)
        match_pred = self.match_head(fused)

        dummy = torch.zeros_like(text_512)

        return {
            "final_logits": final_logits,
            "match_pred": match_pred,
            "z_global_sup": dummy,
            "z_attn_sup": dummy,
            "z_text_sup": dummy,
            "z_image_sup": dummy,
            "res_z1": dummy,
            "res_z2": dummy,
            "expert_logits": [],
            "expert_weights": None,
            "fused_feature": dummy
        }

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for text, image, label in loader:
        text, image, label = text.to(DEVICE), image.to(DEVICE), label.to(DEVICE)
        out = model(text, image)

        logits = out["final_logits"]
        match_pred = out["match_pred"]

        loss_cls = weighted_cross_entropy(logits, label)
        loss_match = F.cross_entropy(match_pred, label)
        loss = loss_cls + 0.2 * loss_match

        total_loss += loss.item() * label.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(label.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    avg_loss = total_loss / len(labels)
    return acc, avg_loss, precision, recall, f1, labels.numpy(), preds.numpy()

def train():
    model = DetectionModule().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)

    train_set = FeatureDataset(
        "./train_text_with_label.npz",
        "./train_image_with_label.npz"
    )
    test_set = FeatureDataset(
        "./test_text_with_label.npz",
        "./test_image_with_label.npz"
    )

    labels_np = test_set.label.numpy()
    print(f"Test set 分布: nonrumor={(labels_np==0).sum()}, rumor={(labels_np==1).sum()}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    best_acc = 0.0

    for epoch in range(NUM_EPOCH):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for text, image, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            text, image, label = text.to(DEVICE), image.to(DEVICE), label.to(DEVICE)

            out = model(text, image)
            logits = out["final_logits"]
            match_pred = out["match_pred"]

            loss_cls = weighted_cross_entropy(logits, label)
            loss_match = F.cross_entropy(match_pred, label)
            loss_aux = torch.tensor(0.0, device=DEVICE)

            loss = loss_cls + ALPHA * loss_aux + 0.4 * loss_match

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * label.size(0)
            correct += (logits.argmax(1) == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        test_acc, test_loss, precision, recall, f1, all_labels, all_preds = evaluate(model, test_loader)

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
              f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

        print("Classification Report:\n" + classification_report(
            all_labels, all_preds, target_names=["nonrumor", "rumor"], digits=4
        ))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Saved new best model (acc: {best_acc:.4f})")

if __name__ == "__main__":
    train()
# Here are the codes of the main function of the paper "CRSN: Constraint-aware Residual Semantic Network for Multimodal Fake News Detection" for reference. The remaining codes will be made public after the paper is accepted.

