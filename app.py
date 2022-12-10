#--------------------------#
#		Librerias		   #
#--------------------------#
import streamlit as st 
import pandas as pd 
import plotly.express as px 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import joblib
from sklearn import metrics
st.set_option('deprecation.showPyplotGlobalUse', False)
#---------------------------------#
# 	Carga de modelos			  #
#---------------------------------#

#reg = joblib.load('modelo_regression')
#classifier = joblib.load('modelo_XGB')

#---------------------------------#
# 	Seleccionar FEATURES		  #
#---------------------------------#
FEATURES_CLASSIF = [#'no_cliente','numero_tarjeta',
'estatus_activo_tdc','proyecto_respiro',
'mcmpl_t1','dias_sin_uso','antiguedad',
'tipo_producto_BLACK','tipo_producto_CLASICA',
'tipo_producto_CORPORATIVA','tipo_producto_FLOTILLA',
'tipo_producto_GOLD','tipo_producto_INFINITE',
'tipo_producto_PLATINUM','tipo_producto_SUPERCASHBACK',
'tipo_producto_SUPERCASHBACK.RD','tarjeta_menos_13_meses',
'tarjeta_castigada','tarjeta_cobrojudicial',
'tarjeta_principal','tarjeta_upgrade',
'tarjeta_visa', 'tarjeta_evertec',
'limite_dop_tdc', 'DIAS_MORA_DOP',
'SALDO_INTERES_DOP', 'SALDO_VENCIDO_DOP',
'limite_usd_tdc','DIAS_MORA_USD',
'SALDO_INTERES_USD','SALDO_VENCIDO_USD',
'limite_mcr','tipo_empleado_banesco_E',
'tipo_empleado_banesco_EX','edad',
'salario_actual','monto_ventas',
'sexo_M','tipo_persona_F','fuente_ingreso_AGOB',
'fuente_ingreso_APRI','fuente_ingreso_INDE',
'fuente_ingreso_SALA','nacionalidad_Dominicano',
'pais_residencia_Dominicano','educacion_LICE',
'educacion_MAES','educacion_PRIM',
'educacion_SECU','educacion_TECN',
'educacion_UNIV','vivienda_propia_S',
'actividad_economica_A','actividad_economica_B',
'actividad_economica_C','actividad_economica_D',
'actividad_economica_E','actividad_economica_F',
'actividad_economica_G','actividad_economica_H',
'actividad_economica_I','actividad_economica_J',
'actividad_economica_K','actividad_economica_L',
'actividad_economica_M','actividad_economica_N',
'actividad_economica_O','actividad_economica_P',
'actividad_economica_Q','trx_DOP_t1','trx_DOP_t2',
'trx_DOP_t3','trx_USD_t1','trx_USD_t2',
'trx_DOP_t4','trx_DOP_t5','trx_DOP_t6',
'trx_DOP_t7','trx_USD_t3','trx_USD_t4',
'trx_USD_t5','trx_DOP_t8','trx_DOP_t9','trx_DOP_t10',
'trx_DOP_t11','trx_DOP_t12','trx_USD_t6','trx_USD_t7',
'trx_USD_t8','trx_USD_t9','trx_USD_t10','trx_USD_t11',
'trx_USD_t12','mfpl_DOP_t1','mfpl_DOP_t2','mfpl_DOP_t3',
'mfpl_USD_t1','mfpl_USD_t2','mfpl_DOP_t4','mfpl_DOP_t5',
'mfpl_DOP_t6','mfpl_DOP_t7','mfpl_USD_t3','mfpl_USD_t4',
'mfpl_USD_t5','mfpl_DOP_t8','mfpl_DOP_t9','mfpl_DOP_t10',
'mfpl_DOP_t11','mfpl_DOP_t12','mfpl_USD_t6','mfpl_USD_t7',
'mfpl_USD_t8','mfpl_USD_t9','mfpl_USD_t10','mfpl_USD_t11',
'mfpl_USD_t12','DOP_t1','DOP_t2','DOP_t3','DOP_t4',
'DOP_t5','DOP_t6','DOP_t7','DOP_t8','DOP_t9','DOP_t10',
'DOP_t11','DOP_t12','USD_t1','USD_t2','USD_t3',
'USD_t4','USD_t5','USD_t6','USD_t7',
'USD_t8','USD_t9',
'USD_t10','USD_t11',
'USD_t12','Prestamo.de.Consumo',
'Certificados.Depositos.a.Plazo',
'Prestamo.Hipotecario',
'Prestamo.Comercial',
'Prestamo.Castigado',
'Prestamo.de.Vehículo',
'Cuenta.de.Ahorro.Corriente'
    #,'NIVEL_DE_RIESGO'
]

TARGET_classifier = ['cancelacion_voluntaria']

FEATURES_REG= [#'no_cliente','numero_tarjeta',
'estatus_activo_tdc','proyecto_respiro',
'mcmpl_t1','dias_sin_uso',
'tipo_producto_BLACK',
'tipo_producto_CLASICA',
'tipo_producto_CORPORATIVA',
'tipo_producto_FLOTILLA',
'tipo_producto_GOLD',
'tipo_producto_INFINITE',
'tipo_producto_PLATINUM',
'tipo_producto_SUPERCASHBACK',
'tipo_producto_SUPERCASHBACK.RD',
'tarjeta_menos_13_meses',
'tarjeta_castigada',
'tarjeta_cobrojudicial',
'tarjeta_principal',
'tarjeta_upgrade',
'tarjeta_visa',
'tarjeta_evertec',
'limite_dop_tdc',
'DIAS_MORA_DOP',
'SALDO_INTERES_DOP',
'SALDO_VENCIDO_DOP',
'limite_usd_tdc',
'DIAS_MORA_USD',
'SALDO_INTERES_USD',
'SALDO_VENCIDO_USD',
'limite_mcr',
'tipo_empleado_banesco_E',
'tipo_empleado_banesco_EX',
'edad',
'salario_actual',
'monto_ventas',
'sexo_M',
'tipo_persona_F',
'fuente_ingreso_AGOB',
'fuente_ingreso_APRI',
'fuente_ingreso_INDE',
'fuente_ingreso_SALA',
'nacionalidad_Dominicano',
'pais_residencia_Dominicano',
'educacion_LICE',
'educacion_MAES',
'educacion_PRIM',
'educacion_SECU',
'educacion_TECN',
'educacion_UNIV',
'vivienda_propia_S',
'actividad_economica_A',
'actividad_economica_B',
'actividad_economica_C',
'actividad_economica_D',
'actividad_economica_E',
'actividad_economica_F',
'actividad_economica_G',
'actividad_economica_H',
'actividad_economica_I',
'actividad_economica_J',
'actividad_economica_K',
'actividad_economica_L',
'actividad_economica_M',
'actividad_economica_N',
'actividad_economica_O',
'actividad_economica_P',
'actividad_economica_Q',
'trx_DOP_t1',
'trx_DOP_t2',
'trx_DOP_t3',
'trx_USD_t1',
'trx_USD_t2',
'trx_DOP_t4',
'trx_DOP_t5',
'trx_DOP_t6',
'trx_DOP_t7',
'trx_USD_t3',
'trx_USD_t4',
'trx_USD_t5',
'trx_DOP_t8',
'trx_DOP_t9',
'trx_DOP_t10',
'trx_DOP_t11',
'trx_DOP_t12',
'trx_USD_t6',
'trx_USD_t7',
'trx_USD_t8',
'trx_USD_t9',
'trx_USD_t10',
'trx_USD_t11',
'trx_USD_t12',
'mfpl_DOP_t1',
'mfpl_DOP_t2',
'mfpl_DOP_t3',
'mfpl_USD_t1',
'mfpl_USD_t2',
'mfpl_DOP_t4',
'mfpl_DOP_t5',
'mfpl_DOP_t6',
'mfpl_DOP_t7',
'mfpl_USD_t3',
'mfpl_USD_t4',
'mfpl_USD_t5',
'mfpl_DOP_t8',
'mfpl_DOP_t9',
'mfpl_DOP_t10',
'mfpl_DOP_t11',
'mfpl_DOP_t12',
'mfpl_USD_t6',
'mfpl_USD_t7',
'mfpl_USD_t8',
'mfpl_USD_t9',
'mfpl_USD_t10',
'mfpl_USD_t11',
'mfpl_USD_t12',
'DOP_t1',
'DOP_t2',
'DOP_t3',
'DOP_t4',
'DOP_t5',
'DOP_t6',
'DOP_t7',
'DOP_t8',
'DOP_t9',
'DOP_t10',
'DOP_t11',
'DOP_t12',
'USD_t1',
'USD_t2',
'USD_t3',
'USD_t4',
'USD_t5',
'USD_t6',
'USD_t7',
'USD_t8',
'USD_t9',
'USD_t10',
'USD_t11',
'USD_t12',
'Prestamo.de.Consumo',
'Certificados.Depositos.a.Plazo',
'Prestamo.Hipotecario',
'Prestamo.Comercial',
'Prestamo.Castigado',
'Prestamo.de.Vehículo',
'Cuenta.de.Ahorro.Corriente',
 #'NIVEL_DE_RIESGO', 
'cancelacion_voluntaria']

TARGET_reg = ['antiguedad']

#------------------------------------#
#	funcionpara data Cleaning        #
#------------------------------------#
def convert_tonumber(df):
    maping =  {'Si':1, 'No':0}
    maping_risk = {'RBAJ': 0 ,'RMED':1 ,'RALT':2}
    df['cancelacion_voluntaria'] = df['cancelacion_voluntaria'].map(maping)
    df['estatus_activo_tdc'] = df['estatus_activo_tdc'].map(maping)
    df['proyecto_respiro'] = df['proyecto_respiro'].map(maping)
    df['tipo_producto_BLACK' ] = df['tipo_producto_BLACK' ].map(maping)
    df['tipo_producto_CLASICA']= df['tipo_producto_CLASICA'].map(maping)
    df['tipo_producto_CORPORATIVA'] = df['tipo_producto_CORPORATIVA'].map(maping)
    df[ 'tipo_producto_FLOTILLA'] = df[ 'tipo_producto_FLOTILLA'].map(maping)
    df['tipo_producto_GOLD'] = df['tipo_producto_GOLD'].map(maping)
    df['tipo_producto_INFINITE'] = df['tipo_producto_INFINITE'].map(maping)
    df['tipo_producto_PLATINUM'] = df['tipo_producto_PLATINUM'].map(maping)
    df['tipo_producto_SUPERCASHBACK'] = df['tipo_producto_SUPERCASHBACK'].map(maping)
    df['tipo_producto_SUPERCASHBACK.RD'] = df['tipo_producto_SUPERCASHBACK.RD'].map(maping) 
    df['tarjeta_menos_13_meses'] = df['tarjeta_menos_13_meses'].map(maping)
    df['tarjeta_castigada'] = df['tarjeta_castigada'].map(maping)
    df['tarjeta_cobrojudicial'] = df['tarjeta_cobrojudicial'].map(maping)
    df['tarjeta_principal'] = df['tarjeta_principal'].map(maping)
    df['tarjeta_upgrade'] = df['tarjeta_upgrade'].map(maping)
    df['tarjeta_visa'] = df['tarjeta_visa'].map(maping)
    df['tarjeta_evertec'] = df['tarjeta_evertec'].map(maping)
    df['tipo_empleado_banesco_E'] = df['tipo_empleado_banesco_E'].map(maping)
    df['tipo_empleado_banesco_EX'] = df['tipo_empleado_banesco_EX'].map(maping) 
    df['sexo_M'] = df['sexo_M'].map(maping)
    df['tipo_persona_F'] = df['tipo_persona_F'].map(maping)
    df['fuente_ingreso_AGOB'] = df['fuente_ingreso_AGOB'].map(maping)
    df['fuente_ingreso_APRI'] = df['fuente_ingreso_APRI'].map(maping)
    df['fuente_ingreso_INDE'] = df['fuente_ingreso_INDE'].map(maping) 
    df['fuente_ingreso_SALA'] = df['fuente_ingreso_SALA'].map(maping) 
    df['nacionalidad_Dominicano'] = df['nacionalidad_Dominicano'].map(maping)
    df['pais_residencia_Dominicano'] = df['pais_residencia_Dominicano'].map(maping)
    df[ 'educacion_LICE'] = df[ 'educacion_LICE'].map(maping)
    df['educacion_MAES'] = df['educacion_MAES'].map(maping)
    df['educacion_PRIM'] = df['educacion_PRIM'].map(maping) 
    df['educacion_SECU'] = df['educacion_SECU'].map(maping) 
    df['educacion_TECN'] = df['educacion_TECN'].map(maping)
    df['educacion_UNIV'] = df['educacion_UNIV'].map(maping)
    df['vivienda_propia_S'] = df['vivienda_propia_S'].map(maping) 
    df['actividad_economica_A'] = df['actividad_economica_A'].map(maping) 
    df['actividad_economica_B'] = df['actividad_economica_B'].map(maping)
    df['actividad_economica_C'] = df['actividad_economica_C'].map(maping)
    df['actividad_economica_D'] = df['actividad_economica_D'].map(maping)
    df['actividad_economica_E'] = df['actividad_economica_E'].map(maping) 
    df['actividad_economica_F'] = df['actividad_economica_F'].map(maping)
    df['actividad_economica_G'] = df['actividad_economica_G'].map(maping) 
    df['actividad_economica_H'] = df['actividad_economica_H'].map(maping)
    df['actividad_economica_I'] = df['actividad_economica_I'].map(maping) 
    df['actividad_economica_J'] = df['actividad_economica_J'].map(maping)
    df['actividad_economica_K'] = df['actividad_economica_K'].map(maping) 
    df['actividad_economica_L'] = df['actividad_economica_L'].map(maping)
    df['actividad_economica_M'] = df['actividad_economica_M'] .map(maping) 
    df['actividad_economica_N'] = df['actividad_economica_N'].map(maping)
    df['actividad_economica_O'] = df['actividad_economica_O'].map(maping) 
    df['actividad_economica_P'] = df['actividad_economica_P'].map(maping)
    df['actividad_economica_Q'] = df['actividad_economica_Q'].map(maping)
    #df['NIVEL_DE_RIESGO'] = df['NIVEL_DE_RIESGO'].map(maping_risk)
    df = df.fillna(0)
    return df

#---------------------------------#
# 		Titulo					  #
#---------------------------------#
st.title('Attrition')


#---------------------------------#
# 	sidebar  					  #
#---------------------------------#
st.sidebar.header('Carge Archivo de excel')

upload_file = st.sidebar.file_uploader('Carge el archivo de excel', type = ['xlsx'])


#---------------------------------#
# 	Contenido  					  #
#---------------------------------#

if upload_file  is not None:
	data = pd.read_excel(upload_file)
	data = convert_tonumber(data)
	#st.write(data.head(10))
	st.write('Cantidad de filas `{}`,  columnas `{}`'.format(data.shape[0], data.shape[1]))
	st.write('Datos Nulos', data.isnull().sum().sum())
	
	#predcciones classificación- probabailidad
	X1 = data[FEATURES_CLASSIF]
	y1 = data[TARGET_classifier]
	y_pred = classifier.predict(X1)	
	y_proba = classifier.predict_proba(data[FEATURES_CLASSIF])


	#predcciones regression prediccion de los días
	X2 = data[FEATURES_REG]
	y2 = data[TARGET_reg]
	y_reg = reg.predict(X2)

	# Metrica de los modelos Accueacy y r2 score
	st.write('Accuracy = ', metrics.accuracy_score(y1,y_pred))
	st.write('R2 score = ', metrics.r2_score(y2,y_reg))

	# Resultados
	data['prediccion'] = y_pred
	data['probabilidad_pred'] = y_proba[:,0]
	data['Predict_days'] = y_reg

	# Creacion de bins
	data['label_pred_days']  = pd.cut(data['Predict_days']
                        , bins=[30, 60, 90, 120, 365]
                        , labels= ['30 a 60', '60 a 90', '90 a 120','120 a 365']
                        )

	# llenar los nulos con mayor a 365
	data['label_pred_days'] = data['label_pred_days'].cat.add_categories('mayor que 365').fillna('mayor que 365')

	
	# Presentación de resultados:
	col1, col2 = st.columns(2)

	col1.write(data['label_pred_days'].value_counts())
	graf = sns.countplot(x= data['label_pred_days'], color = 'green')
	col2.pyplot()
	

	st.download_button(label = "Download Predictions csv file", data = data.to_csv(index =False))

	st.warning('**Created by : `Luis Hernández`**')
	#st.write('Created by: `Luis Hernández`')


else:
	st.write('Created by: `Luis Hernández`')
