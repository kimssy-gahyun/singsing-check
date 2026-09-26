# 라이브러리 불러오기
from tensorflow import keras

# MNIST 데이터셋 불러오기
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""
# MNIST 데이터셋 사이즈 확인
# train 데이터는 60000장, test 데이터는 10000장
# 한 tensor 당 28X28 사이즈
# 각각 0~9 를 표현한 손글씨 이미지
print('train_image.shape : ', train_images.shape) # 60000 X 28 X 28
print('test_image.shape : ', test_images.shape) # 10000 X 28 X 28
print('train_labels.shape : ', train_labels.shape) # 60000
print('test_labels.shape : ', test_labels.shape) # 10000

# train 데이터 중 첫번째 이미지 확인해보기
num = train_images[0] # 28 X 28

for i in range(28) : # 세로길이 28픽셀
  for j in range(28) : # 가로길이 28픽셀
    print('{:4d}'.format(num[i][j]), end='')
    # num[0][0] num[0][1] num[0][2] ... num[0][27] num[1][0] num[1][1] ... num[27][27] 각 픽셀값 출력 (0 ~ 255 사이)
  print() # 줄바꿈

# train 데이터 중 첫번째 이미지의 정답 (label) 확인하기
print('train_labels[0] : ', train_labels[0]) # 5
"""

# ---------------------------------------

# 이미지 데이터 전처리
# 정규화 : 0~255 사이의 값들을 0~1 사이로 조정
train_images_norm = train_images / 1.0
test_images_norm = test_images / 1.0

num = train_images_norm[0] # 28 X 28

"""
for i in range(28) : # 세로길이 28픽셀
  for j in range(28) : # 가로길이 28픽셀
    print('{:5.2f}'.format(num[i][j]), end='') # num[0][0] num[0][1] num[0][2] ... num[0][27] num[1][0] num[1][1] ... num[1][27] ... ... num[27][27] 각 픽셀값 출력 (0 ~ 1 사이)
  print()
"""

# --------------------------------------

# 모델 build (모델의 구조를 정의)
# keras.Sequential : layer 를 순차적으로 쌓겠다.
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Flatten(), # Flatten : tensor 를 vector 화 시키겠다. (입력이 28X28 -> 784)
    keras.layers.Dense(10, activation='softmax') # Dense : 이전 layer와 연결하겠다. (0~9 에 해당하는 출력이 10개 -> softmax 확률표현으로)
])

# 모델 compile (모델을 어떤 규칙으로 학습하고 평가할지 정함)
model.compile(
    optimizer='adam', # optimizer는 'adam' 을 쓸 것 (loss가 감소하도록 가중치를 update하겠다)
    loss='sparse_categorical_crossentropy', # 손실함수 계산은 cross entropy 방식으로 계산할 것 (예측값과 실제 label 사이의 오차 계산)
    metrics=['accuracy'] # 이 모델의 성능은 정확도로 측정할 것
)

# --------------------------------------

# 모델 train (fit)
model.fit(train_images_norm, train_labels, epochs=5, batch_size=32, validation_split=0.1)
# 정규화된 train 데이터와 label 60000개를 이용하여 학습
# epoch = 1 (54000장에 대해 1회만 학습하겠다)
# batch_size = 32 (54000장을 32개씩 넣겠다)
# validation_split = 0.1 (6000장은 validation 용으로 사용하겠다. 실제 학습은 54000장으로 진행)

# --------------------------------------

# 모델 evaluation
model.evaluate(test_images_norm, test_labels) # test 데이터와 레이블 (문제와 답) 을 주고 얼마나 맞추는지 평가
# [0.31276148557662964, 0.9143000245094299] # 10000개의 test 문제 중 약 91.43% 를 맞췄음 (대략 9143/10000 장)
