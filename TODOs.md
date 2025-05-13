# weight loading from previously trained PAT
I was seeing the error:
'''
Traceback (most recent call last):
  File "/home/benle1/Urban-Elements-ReID-Challenge/Code/Files for PAT/train.py", line 74, in <module>
    model = make_model(cfg, modelname=model_name, num_class=num_classes, camera_num=None, view_num=None)
  File "/home/benle1/Urban-Elements-ReID-Challenge/Code/Files for PAT/model/make_model.py", line 345, in make_model
    model = build_part_attention_vit(num_class, cfg, __factory_LAT_type)
  File "/home/benle1/Urban-Elements-ReID-Challenge/Code/Files for PAT/model/make_model.py", line 278, in __init__
    self.base.load_param(self.model_path)
  File "/home/benle1/Urban-Elements-ReID-Challenge/Code/Files for PAT/model/backbones/vit_pytorch.py", line 721, in load_param
    self.state_dict()[k].copy_(v)
KeyError: 'base.cls_token'
'''

When I train the normal embedding, the keys are similar... but have slightly different names.

Then I commented the line 721 (in the original code) of model/backbones/vit_pytorch.py, and substitute by te hardocoded name fix.

The same problem happened in other places, I fixed the same way... I added the comment `# HARDCODED FOR PAT++` where that was needed.

For example, the saved model have `base.` prefixed in the parameter names.

even worse, some weights not even handled at all, like `bottleneck`... Instead of rising an error, I just proceeded.

In a nutshell:
The PAT code assumes we are loading a vanila VIT, loads with those names, then changes those names... Amateurs...and no code is provided to load a model with the NEW names!!!                                     


Just by loading the model this way (without any significant finetuning on generated images), the accuracy is 0.17540 (drop of about 1%).
The 60 epochs with generated images + 60 epochs with real images lead to 0.13972.