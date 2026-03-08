
1. 处理视频文件（保留音频）

python recognize.py --data ./data --source input.mp4 --output output.mp4 --threshold 0.65

--data：数据集文件夹路径。

--source：输入视频文件路径。

--output：输出视频路径（会自动添加原音频）。

--threshold：识别阈值（余弦相似度，默认0.6）。

--skip_frames：跳帧数，提高处理速度（默认2，即每2帧处理1帧）。

2. 摄像头实时识别

python recognize.py --data ./data --camera --source 0

--camera：启用摄像头模式。

--source 0：摄像头设备ID（0表示默认摄像头）。如果不指定 --source 则默认使用0。

可加 --output 保存录像（无声）。




人物             数据个数
-------------------
Neuro-sama |  327个
丛雨             |  402个
电棍	     |  1个
东雪莲 	     | 439个
棍母             |  254个
绫地宁宁 	     |  353个
四季夏目 	     |  3个
永雏塔菲      | 414个