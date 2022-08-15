# import the necessary packages
import base64
import json
import os
import sys
from io import BytesIO

import cv2
import numpy as np
import pymysql
import torch
from flask import Flask, request
from flask_cors import CORS
from matplotlib import pyplot as plt
from torch.autograd import Variable

from data import VOC_CLASSES as labels
from ssd import build_ssd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# create an instance of the Flask class
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的 URL 都允许跨域请求
CORS(app, resources=r'/*')

# create mysql connection
connect = pymysql.connect(host='192.168.1.29',
                          user='root',
                          password='houzhiming',
                          db='bjtu_gpa',
                          charset='utf8')  # 服务器名,账户,密码,数据库名
cur = connect.cursor(cursor=pymysql.cursors.DictCursor)  # 返回json

# 插入SQL
sql_insert = 'insert into T_Images(name,ip,origin_image,annotation_image) values(%s,%s,%s,%s);'
# 查询SQL
sql_select = 'select * from T_Images where name=%s order by id desc;'

'''
    上传图片 & 图像目标检测
    接口地址：/image 
'''


@app.route('/image', methods=['POST'])
def image_detection_save_infos(req=None):
    ''' 1 获取前端数据 '''
    img_name = request.form['name']  # 图片名称
    origin_img_base64_str = request.form['image']  # 原图base64字符串
    ip = request.remote_addr  # 请求ip
    status = 200  # 响应状态码
    message = 'success'  # 响应状态信息
    if len(origin_img_base64_str) == 0:  # 图片非空验证
        status = 500
        message = '请求图片为空'

    ''' 2 将图片字符串转为图片字节流 '''
    img_str = origin_img_base64_str.split(',')[1]  #
    if len(img_str) == 0:
        status = 500
        message = '请求图片为空'

    # 将str转换为bytes,并进行base64解码，得到bytes类型
    filestr = base64.b64decode(img_str.encode())
    npimg = np.frombuffer(filestr, np.uint8)

    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    ''' 3 加载模型 '''
    net = build_ssd('test', 300, 21)  # initialize SSD
    net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    # plt.figure(figsize=(10, 10))
    # plt.imshow(rgb_image)
    # plt.show()

    x = cv2.resize(image, (300, 300)).astype(np.float32)  # 缩放
    x -= (104.0, 117.0, 123.0)  # 减去通道均值,通过减去数据对应维度的统计平均值，来消除公共的部分，以凸显个体之间的特征和差异。
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()  # 将三个通道的顺序恢复为最原始的顺序
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)  # (300,300,3) -> (3,300,300)

    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    ''' 4 模型预测：目标检测得到目标类别和边界框 '''
    y = net(xx)
    detections = y.data

    plt.figure(figsize=(10, 10))  # 创建画板
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # 在画纸上画图 plot the image for matplotlib
    currentAxis = plt.gca()

    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, fontsize=15, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1
    # plt.show() # 展示图片

    ''' 5 将目标检测标注结果图片 转换为 base64字符串 '''
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue())
    annotation_img_str = origin_img_base64_str.split(',')[0] + ',' + str(figdata_png, "utf-8")

    ''' 6 数据库插入当前图片检测记录 '''
    try:
        data = [(img_name, ip, origin_img_base64_str, annotation_img_str)]
        cur.executemany(sql_insert, data)
        cur.connection.commit()  # 事务提交
    except Exception as e:
        status = 500
        message = '服务器异常，请求失败'
        print("插入数据失败:", e)
    else:
        status = 200
        message = '请求成功'
        print("插入数据成功;")

    res = {'status': status, 'message': message}
    return json.dumps(res), 200, {"Content-Type": "application/json"}


@app.route('/annotation', methods=['POST'])
def annotation_infos(req=None):
    img_name = request.form['name']
    ''' 查询图片 '''
    res = {'status': 500, 'message': '服务器异常，查询失败'}
    try:
        data = (img_name)
        cur.execute(sql_select, data)
    except Exception as e:
        print("查询数据失败:", e)
    else:
        print("查询数据成功;")
        sl_res = cur.fetchone()
        if sl_res:
            res = {'status': 200, 'message': 'success', 'id': sl_res['id'], 'name': sl_res['name'],
                   'image': sl_res['origin_image'],
                   'annotation': sl_res['annotation_image']}  # , 'object_infos': sl_res['object_infos']
    return json.dumps(res, cls=JsonEncoder), 200, {"Content-Type": "application/json"}


# json encoder
class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    app.run()  # 运行app
