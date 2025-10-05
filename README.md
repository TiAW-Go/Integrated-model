# Integrated AI-HTS Models — Public Code Snippets & Training Notes

> This document consolidates **hyperparameters** and **training snippets** for the four model families used in our AI‑guided HTS pipeline: **GCN**, **Multi‑level MPNN (GGNN representative)**, **AMPNN (AttentionGGNN representative)**, and **GOM (GCN/GAT/GIN variants with PyTorch Geometric)**. Snippets are adapted from the uploaded sources to form a coherent, public‑facing reference.

---

## 1) GCN (TensorFlow 1.x)

### Key Hyperparameters (defaults from `integrate-model/GCN/train.py`)
```python
# Defaults resolved in train.py
method = "GCN"
prop = "logP"
num_layer = 3
epoch_size = 100
learning_rate = 0.001
decay_rate = 0.95

# Flags (subset)
flags.DEFINE_string('model', method, 'GCN, GCN+a, GCN+g, GCN+a+g')
flags.DEFINE_integer('num_layers', num_layer, '# of hidden layers')
flags.DEFINE_integer('epoch_size', epoch_size, 'Epoch size')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_float('learning_rate', learning_rate, 'LR')
flags.DEFINE_float('decay_rate', decay_rate, 'LR decay')
flags.DEFINE_string('optimizer', 'Adam', 'Adam | SGD | RMSProp')
flags.DEFINE_string('readout', 'atomwise', 'atomwise | graph_gather')
flags.DEFINE_integer('latent_dim', 512, 'latent dim')
```

### Model Construction (from `model/GraphToProperty.py`)
```python
class Graph2Property:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.batch_size = FLAGS.batch_size
        # A: [B, N, N], X: [B, N, F], P: labels
        self.A = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, 50, 50])
        self.X = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, 50, 58])
        self.P = tf.placeholder(dtype=tf.float64, shape=[self.batch_size])
        self.create_network()

    def create_network(self):
        num_layers = self.FLAGS.num_layers
        latent_dim = self.FLAGS.latent_dim
        # Encoder variants: 'GCN', 'GCN+a', 'GCN+g', 'GCN+a+g', 'GGNN'
        if self.FLAGS.model == 'GCN':
            self._X = model.blocks.encoder_gcn(self.X, self.A, num_layers)
        elif self.FLAGS.model == 'GCN+a':
            self._X = model.blocks.encoder_gat(self.X, self.A, num_layers)
        elif self.FLAGS.model == 'GCN+g':
            self._X = model.blocks.encoder_gate(self.X, self.A, num_layers)
        elif self.FLAGS.model == 'GCN+a+g':
            self._X = model.blocks.encoder_gat_gate(self.X, self.A, num_layers)
        elif self.FLAGS.model == 'GGNN':
            self._X = model.blocks.encoder_ggnn(self.X, self.A, num_layers)

        # Readout
        if self.FLAGS.readout == 'atomwise':
            self.Z, self._P = model.blocks.readout_atomwise(self._X, latent_dim)
        else:  # 'graph_gather'
            self.Z, self._P = model.blocks.readout_atomwise(self.X, self._X, latent_dim)

        self.loss = self.calLoss(self.P, self._P)
        self.lr = tf.Variable(0.0, trainable=False)
        self.opt = self.optimizer(self.lr, self.FLAGS.optimizer)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
```

### Training Loop (excerpt from `GCN/train.py`)
```python
def training(model, FLAGS, modelName):
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    decay_rate = FLAGS.decay_rate
    total_iter = 0
    for epoch in range(num_epochs):
        # simple LR schedule per epoch
        model.assign_lr(learning_rate * (decay_rate ** epoch))
        # iterate over databases/shards
        for db_idx in range(FLAGS.num_DB):
            for step in range(FLAGS.unitLen // batch_size):
                A, X, P = loadInputs(FLAGS, step, modelName, FLAGS.unitLen)
                feed = {model.A: A, model.X: X, model.P: P}
                _, loss = model.sess.run([model.opt, model.loss], feed_dict=feed)
                total_iter += 1
```

---

## 2) Multi‑level MPNN (PyTorch) — **GGNN representative**

> Implemented via the MPNN suite in `integrate-model/AMPNN/main`, with flexible hyperparameters exposed by CLI.

### Training Defaults (from `AMPNN/main/train.py`)
```python
# Common args
--epochs=50
--batch-size=50
--learn-rate=1e-5
--loss=CrossEntropy   # or MSE; scoring uses roc-auc or MSE
--score=roc-auc
--cuda                # optional flag

# Model-specific (GGNN)
--message-passes=1
--message-size=25
--msg-depth=2
--msg-hidden-dim=50
--msg-dropout-p=0.0
--gather-width=45
--gather-att-depth=2
--gather-att-hidden-dim=26
--gather-att-dropout-p=0.0
--gather-emb-depth=2
--gather-emb-hidden-dim=26
--gather-emb-dropout-p=0.0
--out-depth=2
--out-hidden-dim=450
--out-dropout-p=0.00463
--out-layer-shrinkage=0.5028
```

### Training Loop (from `AMPNN/main/train.py`)
```python
net = GGNN(...hyperparameters from args...)
optimizer = torch.optim.Adam(net.parameters(), lr=args.learn_rate)
criterion = LOSS_FUNCTIONS[args.loss]

for epoch in range(args.epochs):
    net.train()
    for i_batch, batch in enumerate(train_dataloader):
        if args.cuda:
            batch = [tensor.cuda() for tensor in batch]
        adjacency, nodes, edges, target = batch
        optimizer.zero_grad()
        output = net(adjacency, nodes, edges)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 5.0)
        optimizer.step()

    # evaluation / logging
    with torch.no_grad():
        net.eval()
        LOG_FUNCTIONS[args.logging](
            net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args
        )
```

---

## 3) AMPNN (AttentionGGNN representative, PyTorch)

### Key Hyperparameters (from `AMPNN/main/train.py`)
```python
# Representative defaults for AttentionGGNN
--message-passes=8
--message-size=25
--msg-depth=2
--msg-hidden-dim=50
--msg-dropout-p=0.0

# Attention & edge embeddings
--att-depth=2
--att-hidden-dim=85
--att-dropout-p=0.0
--edge-emb-depth=2
--edge-emb-hidden-dim=105
--edge-emb-dropout-p=0.0

# Readout / gather & output heads
--gather-width=45
--gather-att-depth=2
--gather-att-hidden-dim=45
--gather-att-dropout-p=0.0
--gather-emb-depth=2
--gather-emb-hidden-dim=45
--gather-emb-dropout-p=0.0
--out-depth=2
--out-hidden-dim=450
--out-dropout-p=0.1
--out-layer-shrinkage=0.6
```

> **Training code** is identical in structure to the GGNN loop above; only the constructor and hyperparameters change:

```python
net = AttentionGGNN(...args...)
optimizer = torch.optim.Adam(net.parameters(), lr=args.learn_rate)
# same loop as in Section 2
```

---

## 4) GOM (Graph + Other Mechanisms; PyTorch Geometric)

This folder (`integrate-model/GOM/`) includes **GCN**, **GAT**, **GAT_GCN**, and **GIN** variants implemented with PyTorch Geometric.

### Training Hyperparameters (from `GOM/training.py`)
```python
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][3]  # choose variant
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE  = 128
LR               = 0.001
NUM_EPOCHS       = 1000
LOG_INTERVAL     = 20
device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
```

### Training & Evaluation (from `GOM/training.py`)
```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1,1).float())
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]  Loss: {loss.item():.6f}")

@torch.no_grad()
def predicting(model, device, loader):
    model.eval()
    total_preds, total_labels = torch.Tensor(), torch.Tensor()
    for data in loader:
        data = data.to(device)
        out = model(data)
        total_preds  = torch.cat((total_preds, out.cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1,1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
```

### Data Utilities (from `GOM/utils.py`)
```python
class TestbedDataset(InMemoryDataset):
    def process(self, xd, y, smile_graph):
        # Converts SMILES + labels into PyG Data objects
        data_list = []
        for i in range(len(xd)):
            # build graph, set x/edge_index/edge_attr and y
            data_list.append(DATA.Data(...))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

---

## Notes for Public Release
- The above snippets are **directly adapted** from the provided code to expose key training controls without leaking internal-only paths.
- Replace dataset paths with your public dataset of choice (e.g., ESOL, Tox21) and verify feature extraction (atom/bond featurization) aligns with your data.
- For TF1 GCN, consider porting to TF2/Keras or PyTorch for long-term maintainability.
- For fair comparisons, fix random seeds and log model, data split, and featurization configs.

---

