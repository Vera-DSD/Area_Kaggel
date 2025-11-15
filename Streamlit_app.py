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
    """
    Загружает модель с Google Drive если её нет локально
    """
    model_path = "final_real_estate_pipeline.pkl"
    
    if not os.path.exists(model_path):
        with st.spinner('📥 Скачиваю модель с Google Drive...'):
            try:
                file_id = "1oFv_gIdwuplbBzXIY-3-bJV4FKL6hsyo"
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                st.success("✅ Модель успешно скачана!")
            except Exception as e:
                st.error(f"❌ Ошибка при скачивании модели: {e}")
                return None
    
    try:
        model = joblib.load(model_path)
        st.success("✅ Модель загружена в память!")
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

def prepare_features(input_dict):
    """
    Преобразует ввод пользователя в формат, который ожидает модель
    """
    # Словари для кодирования категориальных признаков
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
            'Спутник': 0,
            'Восточный': 0,
            'Западный': 0,
            'Северный': 0,
            'Южный': 0
        }
    }
    
    # Базовый шаблон со всеми ожидаемыми признаками
    features_template = {
        # Числовые признаки
        'total_area': 0,
        'numbere_of_rooms': 0,
        'ceiling_height': 0,
        'Time_metro': 0,
        'pass_elevators': 0,
        'cargo_elevators': 0,
        
        # Закодированные категориальные признаки
        'renovation_encoded': 0,
        'windows_encoded': 0,
        'children_pets_encoded': 0,
        'balcony_encoded': 0,
        'parking_encoded': 0,
        'bathroom_encoded': 0,
        'metro_encoder': 0,
        
        # One-hot encoded признаки
        'property_Квартира': 0,
        'address_encod': 0
    }
    
    # Заполняем числовые признаки
    for feature in ['total_area', 'numbere_of_rooms', 'ceiling_height', 
                   'Time_metro', 'pass_elevators', 'cargo_elevators']:
        if feature in input_dict:
            features_template[feature] = input_dict[feature]
    
    # Кодируем категориальные признаки
    categorical_mappings = {
        'renovation': 'renovation_encoded',
        'windows': 'windows_encoded',
        'children_pets': 'children_pets_encoded',
        'balcony': 'balcony_encoded',
        'parking': 'parking_encoded',
        'bathroom': 'bathroom_encoded'
    }
    
    for input_key, feature_key in categorical_mappings.items():
        if input_key in input_dict and input_dict[input_key] in encoding_maps[input_key]:
            features_template[feature_key] = encoding_maps[input_key][input_dict[input_key]]
    
    # One-hot encoding для типа недвижимости
    if 'property_type' in input_dict:
        features_template['property_Квартира'] = encoding_maps['property_type'][input_dict['property_type']]
    
    # Кодирование метро
    if 'metro' in input_dict:
        metro_value = input_dict['metro']
        features_template['metro_encoder'] = encoding_maps['metro'].get(metro_value, 0)
    
    return features_template

def create_input_dataframe(prepared_features, model_features=None):
    """
    Создает DataFrame в правильном формате для модели
    """
    df = pd.DataFrame([prepared_features])
    
    if model_features is not None:
        # Добавляем отсутствующие колонки
        missing_cols = set(model_features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        # Убираем лишние колонки и упорядочиваем
        df = df[model_features]
    
    return df

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---

st.title('🏠 Предсказатель цен на недвижимость')
st.markdown('---')

# Загружаем модель
model = load_model()

if model is not None:
    # Получаем ожидаемые признаки модели
    model_features = None
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        st.sidebar.info(f"🎯 Модель ожидает {len(model_features)} признаков")

# Основная форма ввода
if model is not None:
    st.success("✅ Модель готова к работе!")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('📐 Основные характеристики')
        
        total_area = st.slider(
            'Общая площадь (м²)',
            min_value=15.0,
            max_value=200.0,
            value=65.0,
            step=0.5,
            help="От 15 до 200 квадратных метров"
        )

        numbere_of_rooms = st.selectbox(
            "Количество комнат",
            options=[1, 2, 3, 4, 5, 6],
            index=1,
            help="Выберите количество комнат"
        )
        
        ceiling_height = st.slider(
            "Высота потолков (м)",
            min_value=2.3,
            max_value=4.0,
            value=2.7,
            step=0.1,
            help="Высота потолков от 2.3 до 4.0 метров"
        )
        
        Time_metro = st.slider(
            "Время до метро (мин пешком)",
            min_value=1,
            max_value=45,
            value=10,
            help="Время пешком до ближайшей станции метро"
        )
        
        property_type = st.selectbox(
            "Тип недвижимости",
            options=['Квартира', 'Студия', 'Апартаменты', 'Пентхаус'],
            index=0,
            help="Выберите тип недвижимости"
        )
        
        metro = st.selectbox(
            "Район/станция метро",
            options=['Центр', 'Спутник', 'Восточный', 'Западный', 'Северный', 'Южный'],
            index=0,
            help="Выберите район расположения"
        )
    
    with col2:
        st.subheader("🎨 Дополнительные характеристики")
        
        renovation = st.selectbox(
            "Качество ремонта",
            options=['без ремонта', 'косметический', 'евроремонт', 'дизайнерский'],
            index=1,
            help="Выберите качество ремонта"
        )

        balcony = st.selectbox(
            "Балкон/лоджия",
            options=['нет', '1 балкон', '2 балкона', 'лоджия', '2 лоджии'],
            index=1,
            help="Наличие и тип балкона"
        )
        
        windows = st.selectbox(
            "Вид из окон",
            options=['во двор', 'на улицу', 'на улицу и двор'],
            index=1,
            help="Ориентация окон"
        )
        
        parking = st.selectbox(
            "Парковка",
            options=['нет', 'наземная', 'подземная', 'многоуровневая'],
            index=1,
            help="Тип парковки"
        )
        
        bathroom = st.selectbox(
            "Санузел",
            options=['совмещенный', 'раздельный', '2 санузла'],
            index=1,
            help="Тип санузла"
        )
        
        children_pets = st.selectbox(
            "Можно с детьми/животными",
            options=['Можно с животными', 'Можно с детьми', 'Можно с детьми, Можно с животными'],
            index=2,
            help="Ограничения по проживанию"
        )
        
        pass_elevators = st.selectbox(
            "Пассажирских лифтов",
            options=[0, 1, 2, 3],
            index=1,
            help="Количество пассажирских лифтов"
        )
        
        cargo_elevators = st.selectbox(
            "Грузовых лифтов",
            options=[0, 1, 2],
            index=0,
            help="Количество грузовых лифтов"
        )
    
    # Кнопка предсказания
    st.markdown("---")
    if st.button('🎯 Предсказать цену', type="primary", use_container_width=True):
        # Собираем данные
        input_data = {
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
            # Подготавливаем данные
            with st.spinner("🔄 Подготавливаю данные..."):
                prepared_features = prepare_features(input_data)
                input_df = create_input_dataframe(prepared_features, model_features)
            
            # Делаем предсказание
            with st.spinner("🤖 Делаю предсказание..."):
                predicted_price = model.predict(input_df)[0]
            
            # Показываем результаты
            st.markdown("---")
            st.success(f"## 💰 Предсказанная цена: **${predicted_price:,.0f}**")
            
            # Метрики
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📊 Диапазон цен",
                    value=f"${predicted_price * 0.9:,.0f} - ${predicted_price * 1.1:,.0f}",
                    delta="±10%",
                    help="Вероятный диапазон цен"
                )
            
            with col2:
                st.metric(
                    label="🎯 Точность",
                    value="85.6%",
                    help="Средняя точность модели"
                )
            
            with col3:
                st.metric(
                    label="📈 Качество",
                    value="R² = 0.790",
                    help="Коэффициент детерминации"
                )
            
            # Детализация ввода
            with st.expander("📋 Детали введенных данных"):
                st.json(input_data)
                
        except Exception as e:
            st.error(f"❌ Ошибка при предсказании: {e}")
            st.info("💡 Проверьте, что все поля заполнены корректно")

else:
    st.error("""
    ❌ Модель не загружена! 
    
    Возможные причины:
    1. Файл модели не найден
    2. Ошибка при скачивании с Google Drive
    3. Проблемы с подключением к интернету
    """)

# Боковая панель
with st.sidebar:
    st.header("ℹ️ О приложении")
    st.markdown("""
    **ML модель для предсказания цен на недвижимость**
    
    **Характеристики модели:**
    - Алгоритм: Random Forest
    - Точность (MAE): ±$7,412
    - Качество (R²): 0.790
    - Обучена на: 20,000+ объявлений
    
    **Как использовать:**
    1. Заполните все поля формы
    2. Нажмите кнопку "Предсказать цену"
    3. Получите оценку стоимости
    
    **Примечание:** Результат является прогнозом и может отличаться от реальной цены.
    """)
    
    st.markdown("---")
    st.subheader("⚡ Быстрые шаблоны")
    
    template = st.selectbox(
        "Выберите шаблон недвижимости:",
        ["Стандартная 2-комнатная", "Студия в центре", "Премиум 3-комнатная"]
    )
    
    if st.button("Применить шаблон", use_container_width=True):
        if template == "Стандартная 2-комнатная":
            st.session_state.total_area = 65.0
            st.session_state.numbere_of_rooms = 2
            st.session_state.ceiling_height = 2.7
            st.session_state.Time_metro = 15
            st.session_state.property_type = 'Квартира'
            st.session_state.metro = 'Спутник'
            st.session_state.renovation = 'косметический'
        elif template == "Студия в центре":
            st.session_state.total_area = 40.0
            st.session_state.numbere_of_rooms = 1
            st.session_state.ceiling_height = 3.0
            st.session_state.Time_metro = 5
            st.session_state.property_type = 'Студия'
            st.session_state.metro = 'Центр'
            st.session_state.renovation = 'евроремонт'
        elif template == "Премиум 3-комнатная":
            st.session_state.total_area = 95.0
            st.session_state.numbere_of_rooms = 3
            st.session_state.ceiling_height = 3.2
            st.session_state.Time_metro = 10
            st.session_state.property_type = 'Квартира'
            st.session_state.metro = 'Центр'
            st.session_state.renovation = 'дизайнерский'
        
        st.rerun()

# Футер
st.markdown("---")
st.caption("""
📊 *Модель машинного обучения для предсказания цен на недвижимость* • 
R² = 0.790 • MAE = $7,412 • Обновлено: Ноябрь 2024
""")



