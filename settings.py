from pathlib import Path
import os
import logging

name_eval_url = 'https://www.threetong.com/ceming/baziceming/xingmingceshi.php'
save_path = os.path.join(Path(__file__).parent.absolute(), 'data')
assert os.path.isdir(save_path), f'{save_path} not found!'
# create a shared logger
logger = logging.getLogger('shared_logger')
