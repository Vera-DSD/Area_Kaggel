import streamlit as st
import pandas as pd 
import joblib
import numpy as np 
import gdown
import os

# Настройка страницы 
st.set_page_config(
    page_title="Предсказатель цен на недвижимость",
    page_icon="🏠", 
    layout='wide'
)

@st.cache_resource
def load_model():
    model_path = "final_real_estate_pipeline.pkl"
    
    if not os.path.exists(model_path):
        with st.spinner('📥 Скачиваю модель с Google Drive...'):
            file_id = "1oFv_gIdwuplbBzXIY-3-bJV4FKL6hsyo"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Модель скачана!")
    
    try:
        model = joblib.load(model_path)
        st.success("✅ Модель загружена в память!")
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

def prepare_features(input_dict):
    """
    Преобразует сырые данные в формат, который ожидает модель
    """
    # Создаем словарь для маппинга значений
    encoding_maps = {
        'renovation': {
            'без ремонта': 0,
            'косметический': 1,
            'евроремонт': 2,
            'дизайнерский': 3
        },
        'windows': {
            'во двор': 0,
            'на улицу': 1,
            'на улицу и двор': 2
        },
        'children_pets': {
            'Можно с животными': 0,
            'Можно с детьми': 1,
            'Можно с детьми, Можно с животными': 2
        },
        'balcony': {
            'нет': 0,
            '1 балкон': 1,
            '2 балкона': 2,
            'лоджия': 3,
            '2 лоджии': 4
        },
        'parking': {
            'нет': 0,
            'наземная': 1,
            'подземная': 2,
            'многоуровневая': 3,
            'на крыше': 4
        },
        'bathroom': {
            'совмещенный': 0,
            'раздельный': 1,
            '2 санузла': 2
        },
        'property_type': {
            'Квартира': 1,
            'Студия': 0,
            'Апартаменты': 0,
            'Пентхаус': 0
        },
        'metro': {
            'Центр': 1,
            'Спутник': 0
            # Добавьте другие станции метро
        }
    }
    
    # Создаем базовый DataFrame с нулями для всех ожидаемых фич
    expected_features = {
        'renovation_encoded': 0,
        'windows_encoded': 0,
        'children_pets_encoded': 0,
        'balcony_encoded': 0,
        'address_encod': 0,  # По умолчанию 0
        'property_Квартира': 0,
        'metro_encoder': 0,
        'parking_encoded': 0,
        'bathroom_encoded': 0,
        'total_area': 0,
        'numbere_of_rooms': 0,
        'ceiling_height': 0,
        'Time_metro': 0,
        'pass_elevators': 0,
        'cargo_elevators': 0
    }
    
    # Заполняем числовые фичи
    numeric_features = ['total_area', 'numbere_of_rooms', 'ceiling_height', 
                       'Time_metro', 'pass_elevators', 'cargo_elevators']
    
    for feature in numeric_features:
        if feature in input_dict:
            expected_features[feature] = input_dict[feature]
    
    # Кодируем категориальные фичи
    if 'renovation' in input_dict and input_dict['renovation'] in encoding_maps['renovation']:
        expected_features['renovation_encoded'] = encoding_maps['renovation'][input_dict['renovation']]
    
    if 'windows' in input_dict and input_dict['windows'] in encoding_maps['windows']:
        expected_features['windows_encoded'] = encoding_maps['windows'][input_dict['windows']]
    
    if 'children_pets' in input_dict and input_dict['children_pets'] in encoding_maps['children_pets']:
        expected_features['children_pets_encoded'] = encoding_maps['children_pets'][input_dict['children_pets']]
    
    if 'balcony' in input_dict and input_dict['balcony'] in encoding_maps['balcony']:
        expected_features['balcony_encoded'] = encoding_maps['balcony'][input_dict['balcony']]
    
    if 'parking' in input_dict and input_dict['parking'] in encoding_maps['parking']:
        expected_features['parking_encoded'] = encoding_maps['parking'][input_dict['parking']]
    
    if 'bathroom' in input_dict and input_dict['bathroom'] in encoding_maps['bathroom']:
        expected_features['bathroom_encoded'] = encoding_maps['bathroom'][input_dict['bathroom']]
    
    # One-hot encoding для property_type
    if 'property_type' in input_dict:
        expected_features['property_Квартира'] = encoding_maps['property_type'][input_dict['property_type']]
    
    # Кодирование метро (упрощенное)
    if 'metro' in input_dict:
        metro_name = input_dict['metro']
        if metro_name in encoding_maps['metro']:
            expected_features['metro_encoder'] = encoding_maps['metro'][metro_name]
        else:
            # Для неизвестных станций используем значение по умолчанию
            expected_features['metro_encoder'] = 0
    
    # address_encod - установим в 0 если не используется
    expected_features['address_encod'] = 0
    
    return expected_features

# Основной заголовок
st.title('Предсказатель цен на недвижимость')
st.markdown('---')

# Загружаем модель 
model = load_model()

if model is not None:
    try:
        # Посмотрите, какие фичи ожидает модель
        if hasattr(model, 'feature_names_in_'):
            st.write("Ожидаемые фичи:", list(model.feature_names_in_))
        st.success("✅ Модель готова к работе!")
    except Exception as e:
        st.error(f"❌ Ошибка инициализации модели: {e}")

if model is not None:
    st.success("Модель успешно загружена!")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Основные характеристики')
        total_area = st.number_input(
            'Общая площадь(м2)',
            min_value=10.0,
            max_value=500.0,
            value=65.0,
            step=0.5
        )

        numbere_of_rooms = st.selectbox(
            "Количество комнат",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        
        ceiling_height = st.number_input(
            "Высота потолков (м)",
            min_value=2.0,
            max_value=5.0,
            value=2.7,
            step=0.1
        )
        
        Time_metro = st.slider(
            "Время до метро (мин)",
            min_value=1,
            max_value=60,
            value=15
        )
        
        property_type = st.selectbox(
            "Тип недвижимости",
            options=['Квартира', 'Студия', 'Апартаменты', 'Пентхаус']
        )
        
        metro = st.text_input(
            "Станция метро",
            value="Центр"
        )
    
    with col2:
        st.subheader("🎨 Дополнительные характеристики")
        
        renovation = st.selectbox(
            "Ремонт",
            options=['без ремонта', 'косметический', 'евроремонт', 'дизайнерский']
        )

        balcony = st.selectbox(
            "Балкон",
            options=['нет', '1 балкон', '2 балкона', 'лоджия', '2 лоджии']
        )
        
        windows = st.selectbox(
            "Окна",
            options=['во двор', 'на улицу', 'на улицу и двор']
        )
        
        parking = st.selectbox(
            "Парковка",
            options=['нет', 'наземная', 'подземная', 'многоуровневая', 'на крыше']
        )
        
        bathroom = st.selectbox(
            "Санузел",
            options=['совмещенный', 'раздельный', '2 санузла']
        )
        children_pets = st.selectbox(
            "Можно с детьми/животными",
            options=['Можно с животными', 'Можно с детьми', 'Можно с детьми, Можно с животными']
        )
        
        pass_elevators = st.selectbox(
            "Пассажирских лифтов",
            options=[0, 1, 2, 3, 4, 5]
        )
        
        cargo_elevators = st.selectbox(
            "Грузовых лифтов",
            options=[0, 1, 2, 3, 4]
        )
    
    # Поле для удобств
    st.subheader("🛋️ Удобства")
    amenities = st.text_area(
        "Удобства (перечислите через запятую)",
        value="Мебель на кухне, Ванна, Стиральная машина, Кондиционер",
        help="Например: Мебель, Кондиционер, Холодильник, Посудомоечная машина"
    )

    # Кнопка предсказаний 
    if st.button('🎯 Предсказать цену', type= "primary"):
        # Собираем сырые данные
        features_dict = {
            'total_area': total_area,
            'numbere_of_rooms': numbere_of_rooms,
            'ceiling_height': ceiling_height,
            'Time_metro': Time_metro,
            'property_type': property_type,
            'metro': metro,
            'renovation': renovation,
            'balcony': balcony,
            'windows': windows,
            'parking': parking,
            'bathroom': bathroom,
            'children_pets': children_pets,
            'pass_elevators': pass_elevators,
            'cargo_elevators': cargo_elevators
        }
        
        try:
            # Преобразуем данные в формат модели
            prepared_features = prepare_features(features_dict)
            
            # Создаем DataFrame в правильном порядке
            if hasattr(model, 'feature_names_in_'):
                # Упорядочиваем колонки как ожидает модель
                input_df = pd.DataFrame([prepared_features])[model.feature_names_in_]
            else:
                input_df = pd.DataFrame([prepared_features])
            
            # Показываем отладочную информацию
            with st.expander("📊 Отладочная информация"):
                st.write("Подготовленные фичи:", prepared_features)
                st.write("DataFrame для предсказания:", input_df)
            
            # Делаем предсказание
            predicted_price = model.predict(input_df)[0]
            
            # Показываем результат
            st.markdown("---")
            st.success(f"## 🏠 Предсказанная цена: ${predicted_price:,.0f}")
            
            # Дополнительная информация
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                st.metric(
                    label="📊 Диапазон цен",
                    value=f"${predicted_price * 0.9:,.0f} - ${predicted_price * 1.1:,.0f}",
                    help="Вероятный диапазон ±10%"
                )

            with col_result2:
                st.metric(
                    label="🎯 Точность модели",
                    value="85.6%",
                    help="Средняя точность на тестовых данных"
                )
            
            with col_result3:
                st.metric(
                    label="📈 Качество модели",
                    value="R² = 0.790",
                    help="Коэффициент детерминации"
                )
                
        except Exception as e:
            st.error(f"❌ Ошибка при предсказании: {e}")
            st.info("Проверьте отладочную информацию выше для диагностики проблемы")

# ... остальной код (sidebar и футер) остается без изменений
    # Футер
        st.markdown("---")
        st.markdown(
    "📊 *Модель машинного обучения для предсказания цен на недвижимость* • "
    "R² = 0.790 • MAE = $7,412")



