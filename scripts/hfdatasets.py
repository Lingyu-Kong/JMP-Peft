# %%
import ll
import rich

ll.pretty()
# %%
from datasets import Dataset
from jmppeft.configs.pretrain.tasks import tasks_config_perlmutter_
from jmppeft.tasks.pretrain import module as M

config = M.PretrainConfig.draft()
tasks_config_perlmutter_(config)
dataset_config = config.tasks[0].train_dataset
rich.print(dataset_config)

# %%
