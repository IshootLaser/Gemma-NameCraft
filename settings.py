from pathlib import Path
import os
import logging

name_eval_url = 'https://www.threetong.com/ceming/baziceming/xingmingceshi.php'
save_path = os.path.join(Path(__file__).parent.absolute(), 'data')
# create a shared logger
logger = logging.getLogger('shared_logger')
