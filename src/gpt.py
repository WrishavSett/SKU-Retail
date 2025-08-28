t_itemdesc = "SOFTY BABY PANT/M CRAWLER/4 PICES/PPH"
m_sku_list = {1: "SOFTY BABY PANT-SMALL(S) PPH 4 NOS", 
              2: "SOFTY BABY PANT-MEDIUM(M) PPH 4 NOS", 
              3: "SOFTY BABY PANT-LARGE(L) PPH 4 NOS", 
              4: "TENDER TOUCH BABY PULL UP PANTS-MEDIUM(M) PPH 5 NOS", 
              5: "TENDER TOUCH BABY PULL UP PANTS-SMALL(S) PPH 5 NOS"}

import re
import json
import ollama

prompt = f"""
Item Description: {t_itemdesc}
SKU List: {json.dumps(m_sku_list, indent=2)}
"""