# AI_Image_Search_Engine 🖼️🔍

## Description
이미지 입력을 받아 해당 이미지와 가장 유사한 이미지를 벡터 검색을 통해 찾고, 원본 이미지와 유사한 이미지들을 시각적으로 비교하여 보여줍니다. 검색된 각 이미지에는 유사도 정보가 함께 제공됩니다.

<br>

## Installation and Execution

### 실행 방법
이미지 검색 엔진을 실행하려면 아래 단계를 따르세요:

1. `ai_image_search_engine.py` 파일을 실행하여 이미지 검색 엔진을 시작합니다.
2. 검색하고자 하는 이미지를 입력하여 시스템에 로드합니다.
3. 시스템이 해당 이미지를 벡터로 변환한 후, 데이터베이스에 있는 이미지들과 비교하여 유사한 이미지를 검색합니다.
4. 결과로 검색된 유사한 이미지를 원본 이미지와 함께 비교하여 시각적으로 확인할 수 있습니다.

프로그램을 실행하려면 다음 명령어를 사용하세요:

```bash
python ai_image_search_engine.py
```
<br>

## 주요 함수 설명

#### 이미지 임베딩 생성 및 변환: `ai_image_search_engine.py`
```python
# 이미지를 벡터로 변환하는 함수
img_tensor = feature_extractor(images=img, return_tensors="pt").to("cuda")
outputs = model(**img_tensor)
embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()
```
이 코드는 `Hugging Face`의 **ViTFeatureExtractor**와 **ViTModel**을 사용하여 입력된 이미지를 텐서로 변환한 후, **비전 트랜스포머(ViT)** 모델을 통해 이미지를 임베딩(벡터화)하는 과정입니다. 
1. `feature_extractor`를 통해 이미지를 텐서 형식으로 변환합니다.
2. 변환된 이미지를 **ViTModel**에 전달하여 임베딩을 생성합니다.
3. 최종적으로 모델의 출력에서 `pooler_output`을 추출하여, 텐서 데이터를 넘파이 배열로 변환한 후 1차원 벡터로 변환합니다.

#### 모든 이미지 벡터화: `ai_image_search_engine.py`
```python
for i, img_path in enumerate(tqdm(img_list)):
    img = Image.open(img_path)
    img_tensor = feature_extractor(images=img, return_tensors="pt").to("cuda")
    outputs = model(**img_tensor)
    embedding = outputs.pooler_output.detach().cpu().numpy().squeeze().tolist()
    embeddings.append(embedding)
    metadatas.append({"uri": img_path, "name": cls})
    ids.append(str(i))
```
이 코드는 이미지 데이터셋에 있는 모든 이미지를 임베딩하여 데이터베이스에 저장하는 과정입니다.
1. `tqdm` 모듈을 사용하여 이미지 경로 리스트(`img_list`)의 진행 상황을 표시합니다.
2. 각 이미지 파일을 열고 `feature_extractor`를 통해 이미지를 벡터로 변환한 후, 모델을 사용해 임베딩을 추출합니다.
3. 임베딩, 메타데이터(이미지 경로와 클래스), 그리고 고유 ID를 각각 리스트에 저장합니다.

#### 벡터 검색 수행 및 쿼리: `ai_image_search_engine.py`
```python
query_result = collection.query(
    query_embeddings=[test_embedding],
    n_results=3,
)
```
이 함수는 `Chroma DB`를 활용하여 검색 엔진을 구현하는 과정입니다. 쿼리 이미지를 임베딩한 후, 데이터베이스에 저장된 임베딩들과 비교하여 가장 유사한 이미지를 검색합니다.
1. `query_embeddings`로 검색할 이미지의 임베딩을 입력합니다.
2. `n_results`는 검색된 유사한 이미지의 결과 수를 지정합니다.

#### 검색 결과 시각화: `ai_image_search_engine.py`
```python
fig, axes = plt.subplots(1, 3, figsize=(16, 10))
for i, metadata in enumerate(query_result["metadatas"][0]):
    distance = query_result["distances"][0][i]
    axes[i].imshow(Image.open(metadata["uri"]))
    axes[i].set_title(f"{metadata['name']}: {distance:.2f}")
    axes[i].axis("off")
```
이 코드는 검색된 유사한 이미지들을 시각적으로 표시하는 부분입니다.
1. `matplotlib`의 `subplots`를 사용해 결과 이미지를 그릴 서브플롯을 생성합니다.
2. 검색된 각 이미지의 메타데이터를 기반으로 이미지를 표시하며, 해당 이미지와의 유사도 거리를 이미지 제목으로 설정합니다.
3. 이미지를 깔끔하게 출력하기 위해 축을 숨깁니다.

<br>

## 참고사항: Chroma DB와 Hugging Face DINO 모델
- **Chroma DB**는 이미지 임베딩을 저장하고 유사도를 계산하기 위한 벡터 데이터베이스입니다.
- **Hugging Face DINO**는 이미지에서 특징을 추출하는 비전 트랜스포머 모델로, 이미지 임베딩을 생성하는 데 사용됩니다.

<br>

## Contributor
- kks0507

<br>

## License
This project is licensed under the MIT License.

## Repository
코드 및 프로젝트의 최신 업데이트는 [여기](https://github.com/kks0507/AI_Image_Search_Engine.git)에서 확인할 수 있습니다.

