# Walmart Sales Forecasting - ექსპერიმენტები და რეპორტი

## 1. პროექტის მიმოხილვა და მონაცემების დამუშავება

ჩვენი ამოცანა იყო Walmart-ის 45 სხვადასხვა მაღაზიის და მათი დეპარტამენტების ყოველკვირეული გაყიდვების პროგნოზირება (Time-Series Forecasting). 
ამოცანის მთავარი სირთულე არის ის, რომ გვაქვს ათასობით დამოუკიდებელი დროითი სერია (store-dept წყვილები), მკვეთრი სეზონურობა და არასტანდარტული spike-ები დღესასწაულებზე.

**მონაცემები და Feature Engineering:**

გაერთიანების შემდეგ (`train`, `test`, `features`, `stores`), საერთო pipeline-ში (`feature_pipeline.py`) დავამატეთ შემდეგი ჯგუფის ფიჩერები:

- **Markdown-ები:** `MarkDown1-5` სვეტების NaN მნიშვნელობები ჩავანაცვლეთ 0-ით, დავამატეთ `markdown_total` (ჯამი) და `n_active_markdowns` (რამდენი markdown არის აქტიური კონკრეტულ კვირაში).
- **სერიის გაფანტულობა (sparsity):** თითოეული (Store, Dept) წყვილისთვის დავთვალეთ `n_weeks` (რამდენი კვირის ისტორია აქვს train-ში) და `is_sparse` ფლეგი (< 52 კვირა).
- **კალენდარული ფიჩერები:** `week_of_year`, `month`, `year`, `days_to_nearest_holiday` (უახლოეს Super Bowl/Labor Day/Thanksgiving/Christmas თარიღამდე დღეების რაოდენობა), და `is_week_after_thanksgiving` (სპეციალური ფლეგი, რადგან Thanksgiving-ის მომდევნო კვირას ხშირად სპეციფიკური გაყიდვების ცვლილება აქვს).
- **Lag და Rolling ფიჩერები:** `lag_1-4`, `lag_52`, `lag_53`, `rolling_mean_4/8`, `rolling_std_4/8`, და `yoy_growth` (`lag_1`-ისა და `lag_53`-ის შეფარდება).
- **ჯგუფური აგრეგატები:** `dept_avg_sales` და `store_avg_sales` — თითოეული დეპარტამენტის/მაღაზიის საშუალო გაყიდვები (გამოთვლილი მხოლოდ train მონაცემებზე, რათა თავიდან ავიცილოთ leakage).
- **Store Type-ის დაშიფვრა:** `type_ordinal` — Type A/B/C გადავიყვანეთ ორდინალურ სკალაზე (A=3, B=2, C=1), რადგან მაღაზიის ტიპი პირდაპირ კავშირშია მაღაზიის ზომასთან/გაყიდვების მოცულობასთან.

**მნიშვნელოვანი შენიშვნა მოდელების მიხედვით:** ეს pipeline ქმნის საერთო ფიჩერებს, რომლებიდანაც თითოეულმა მოდელმა შემდეგ თავად შეარჩია მისთვის შესაბამისი ქვეჯგუფი. მაგალითად, XGBoost-ისთვის მოკლევადიანი lag-ები (`lag_1-4`, `rolling_mean/std`) ჩამოვაგდეთ feature selection ეტაპზე, რადგან ისინი ვერ არსებობს ტესტ სეტის უმეტესი ნაწილისთვის.

**Train / Val Split (Time-based):**
Time-series ამოცანებში მონაცემების random split-ით დაყოფა არ შეიძლება - თუ მომავლის მონაცემები train-ში მოხვდება, მოდელი "დაიზეპირებს". 
ამიტომ გამოვიყენეთ **time-based split**. Train სეტის ბოლო 12 კვირა გადავდეთ validation-ისთვის.

---

## 2. მოდელების არქიტექტურა და ექსპერიმენტები 

მთლიანი პროცესი (preprocessing, feature selection, HPO) დავლოგეთ **MLflow**-ში (DagsHub-ზე). 

### 2.1 Deep Learning (N-BEATS, PatchTST, DLinear)

Deep Learning მოდელებისთვის მონაცემები დავაჯგუფეთ `store-dept` წყვილებად და გადავაქციეთ 1D ვექტორებად (lookback=12, horizon=1).

**N-BEATS:**
- იყენებს forward და backward residual ბლოკებს. ვცადეთ სხვადასხვა stack და block კონფიგურაციები (მაგ. 3 stacks, 512 hidden). 
- **შედეგი:** ჩვენს 1D დროითი სერიების მოდელებს შორის საუკეთესო შედეგი აჩვენა. HPO ეტაპზე საუკეთესო კომბინაციამ `WMAE = 1263.22` დადო. კარგად დაიჭირა pattern-ები და აჯობა უფრო ახალ არქიტექტურებს (PatchTST).

**PatchTST (Transformer):**
- **არქიტექტურა:** ეს მოდელი დროით სერიას ჭრის პატარა patch-ებად (`patch_len=4`). ეს ეხმარება ტრანსფორმერს ლოკალური სემანტიკის შენარჩუნებაში.
- **პრობლემა (NaN Explosion):** ტრენინგის დასაწყისში PyTorch-ის Transformer-ში attention წონები იბერებოდა და გვიბრუნებდა NaN-ებს. ამის მოსაგვარებლად მოდელში დავამატეთ **Instance Normalization** (სერიის mean-ის და std-ის გამოთვლა წინასწარ), რამაც ტრენინგი დაასტაბილურა.
- **Overfitting:** თავიდან ავიღეთ დიდი მოდელი (`d_model=128`, 4 layers). Train loss ძალიან სწრაფად ჩამოვიდა (1600-ებზე), მაგრამ val loss გაჩერდა - ანუ მოდელი იზეპირებდა train-ს. რადგან ჩვენი დასატრენინგებელი dataset არ არის საკმარისად დიდი ასეთი ტრანსფორმერისთვის, შევამცირეთ მოდელი (`d_model=64`, 2 layers), რამაც უკეთესი განზოგადება მოგვცა და საბოლოოდ `WMAE = 1305.83` მივიღეთ.

**DLinear:**
- ის სერიას Moving Average ფენით შლის Trend და Seasonal კომპონენტებად და ცალ-ცალკე ატარებს Linear projection-ში.
- **შედეგი:** ძალიან სწრაფად იტრენინგა, თუმცა WMAE მეტრიკით ყველაზე ცუდი შედეგი დადო Deep Learning მოდელებს შორის (საუკეთესო HPO `WMAE = 1515.11`). სავარაუდოდ მისი ზედმეტი სიმარტივე ვერ იჭერს Walmart-ის კომპლექსურ სეზონურობას.

### 2.2 Foundation Models (TimesFM)

- ასევე გავტესტეთ Google-ის `timesfm-2.5-200m-pytorch`. 
- ეს მოდელი არის Zero-shot. საერთოდ არ დაგვიტრეინინგებია ჩვენს train set-ზე. პირდაპირ გადავეცით ისტორია  და ვთხოვეთ 12 კვირის პროგნოზი.
- **ექსპერიმენტი:** ვცადეთ სხვადასხვა context length (52 კვირა, 104 კვირა, სრული ისტორია). 
- **შედეგი:** სრული context length-ით `WMAE = 1252.85` მივიღეთ, რაც საკმაოდ კარგი შედეგია Zero-shot მოდელისთვის. ფაქტობრივად, ყოველგვარი წვრთნისა და exogenous ფიჩერების გარეშე, თითქმის ყველა სხვა მოდელს აჯობა.

### 2.3 Tree-Based Baseline (LightGBM)

- აქ სერიები 1D ვექტორების ნაცვლად დავტოვეთ tabular ფორმატში. 
- Walmart-ის მონაცემებში გვაქვს ბევრი exogenous ცვლადი (მარკდაუნები, ტემპერატურა, უმუშევრობა, მაღაზიის ზომა). Tree-based მოდელები ასეთ tabular ფიჩერებს ბევრად უკეთ იყენებენ.
- LightGBM-ში objective-ად ავიღეთ `regression_l1` (MAE), რადგან WMAE პირდაპირ კავშირშია L1 loss-თან.
- **შედეგი:** Full features სეტმა უკეთესი შედეგი აჩვენა, ვიდრე feature selection-ის მერე დარჩენილმა პატარა სეტმა (`val WMAE = 1251.01` vs `1256.77`). HPO ეტაპზე `num_leaves` და `learning_rate` გავტუნეთ.

---

## 3. შეფასება (Evaluation)

შეჯიბრის ოფიციალური მეტრიკა არის **WMAE**.

MLflow-ში დალოგილი საუკეთესო ვერსიების შედეგები (Val WMAE):

| მოდელი | მიდგომა | Val WMAE |
|--------|---------|----------|
| **LightGBM** | Tabular / Tree-Based | **1251.01** |
| **TimesFM** | Zero-Shot Foundation | **1252.85** |
| N-BEATS | 1D / Deep Learning | 1263.22 |
| PatchTST | 1D / Transformer | 1305.83 |
| DLinear | 1D / Linear | 1515.11 |
| *(XGBoost)* | *(დასამატებელია)* | - |
| *(Prophet)* | *(დასამატებელია)* | - |
| *(SARIMA)* | *(დასამატებელია)* | - |

*(ჩემი მოდელების ანალიზი):*
როგორც ტრენინგის ლოგებიდან ჩანს, LightGBM-მა (1251) და TimesFM-მა (1252) საუკეთესო შედეგები აჩვენეს. 
- **LightGBM** ის კარგად იმიტომ მუშაობს რომ exogenous ფიჩერების (Store Size, Markdown-ები) Tabular ფორმატში გადამუშავება მისთვის მარტივია, მაშინ როცა 1D Deep Learning მოდელებს (PatchTST, DLinear) უჭირთ ამ ინფორმაციის ისტორიასთან ერთად დასწავლა.
- **TimesFM** - როგორც ვახსენეთ, საკმაოდ ძლიერი zero-shot მოდელია რომელმაც წვრთნის გარეშეც კი შეძლო ძალიან კარგი შედეგის დადება.

---

## 4. Model Registry და კოდის სტრუქტურა

ექსპერიმენტების დასრულების შემდეგ, საუკეთესო შედეგების მქონე მოდელები დავარეგისტრირეთ **MLflow Model Registry**-ში. 

ჩვენი საბოლოო inference ლოგიკა მოცემულია `model_inference.ipynb` ფაილში: პროდაქშენ მოდელი იტვირთება პირდაპირ MLflow-დან, ხდება მონაცემების (`test.csv`) გადაცემა და საბოლოო `submission.csv` ფაილის გენერაცია Kaggle-ზე ასატვირთად.

```
ml_final_project/
├── data/                       # raw და processed მონაცემები
├── src/                        # custom pipeline (PyTorch/LGBM) კლასები
├── notebooks/                  
│   ├── data_eda.ipynb              # EDA და Feature Engineering 
│   ├── model_experiment_LightGBM.ipynb
│   ├── model_experiment_PatchTST.ipynb
│   ├── model_experiment_DLinear.ipynb
│   ├── model_experiment_NBEATS.ipynb
│   ├── model_experiment_TimesFM.ipynb
│   └── model_inference.ipynb       # საბოლოო predict Model Registry-დან
└── README.md                   
```

---

## 5. საბოლოო დასკვნა
დასამატებელია
