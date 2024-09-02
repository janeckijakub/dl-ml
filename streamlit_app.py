import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler



st.title('❤ My Machine Learning')

st.info('This is a ML app, building a model based on data')


with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.dataframe(df, use_container_width=True)

    st.write('**X**')
    X = df.drop('species', axis=1)
    st.dataframe(X, use_container_width=True)

    st.write('**y**')
    y = df.species
    y = pd.DataFrame(y)
    st.dataframe(y, use_container_width=True)

with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
    st.scatter_chart(data=df, x='bill_length_mm', y='flipper_length_mm', color='species')

# Sidebar input features
with st.sidebar:
    st.header('Input features')

    idx = st.slider('Index',0,len(X)-1)

    st.write(df.iloc[idx]['species'])

    islands_name = df.island.unique()
    sex_name = df.sex.unique()

    island = st.selectbox('Choose an island:', islands_name, index=islands_name.tolist().index(df.iloc[idx]['island']))
    sex = st.selectbox('Choose a gender:', sex_name, index=sex_name.tolist().index(df.iloc[idx]['sex']))

    bill_length = [df.bill_length_mm.min(), df.bill_length_mm.max()]
    bill_length_mm = st.slider('bill_length_mm', bill_length[0], bill_length[1], value=df.iloc[idx]['bill_length_mm'] )

    bill_depth = [df.bill_depth_mm.min(), df.bill_depth_mm.max()]
    bill_depth_mm = st.slider('bill_depth_mm', bill_depth[0], bill_depth[1], value=df.iloc[idx]['bill_depth_mm'])

    flipper_length = [df.flipper_length_mm.min(), df.flipper_length_mm.max()]
    flipper_length_mm = st.slider('flipper_length_mm', flipper_length[0], flipper_length[1], value=df.iloc[idx]['flipper_length_mm'])

    body_mass = [df.body_mass_g.min(), df.body_mass_g.max()]
    body_mass_g = st.slider('body_mass_g', body_mass[0], body_mass[1], value=df.iloc[idx]['body_mass_g'])

    

with st.expander('Input Features'):
    # Create DataFrame for input features
    st.write('**Input penguin**')
    input_df = pd.DataFrame([{
        'island': island, 
        'bill_length_mm': bill_length_mm, 
        'bill_depth_mm': bill_depth_mm, 
        'flipper_length_mm': flipper_length_mm, 
        'body_mass_g': body_mass_g, 
        'sex': sex
    }])
    st.dataframe(input_df, use_container_width=True)

    st.write('**Combined DataFrame**')
    input_penguins = pd.concat([input_df, X], axis=0)
    st.dataframe(input_penguins, use_container_width=True)
    
with st.expander('Data Preparation'):
    # Encode categorical variables in X
    categorical_columns_X = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns_X)
    st.write('**X_encoded**')
    st.dataframe(X_encoded, use_container_width=True)
    
    # Encode input_row in the same way as X_encoded
    input_row = pd.get_dummies(input_df, columns=categorical_columns_X)
    
    input_row_index = X_encoded.loc[idx]
    
    # Align input_row with X_encoded
    input_row = input_row.reindex(columns=X_encoded.columns, fill_value=0)
    st.write('**Encoded Input Row**')
    st.dataframe(input_row, use_container_width=True)

    # Encode y (target variable)
    if 'species' in y.columns:
        codes, uniques = pd.factorize(y['species'])
        y_encoded = pd.DataFrame({'species_encoded': codes}).reset_index(drop=True)

        st.write('**Encoded Target**')
        uniques_df = pd.DataFrame(uniques, columns=['species'])
        st.dataframe(uniques_df, use_container_width=True)
        st.write('**y_encoded**')
        st.dataframe(y_encoded, use_container_width=True)
    else:
        st.error('The column "species" was not found in y.')

with st.expander('Scaled data'):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    input_row_scaled = scaler.transform(input_row)
    input_row_scaled = pd.DataFrame(input_row_scaled, columns=X_encoded.columns)
    st.dataframe(X_scaled, use_container_width=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded['species_encoded'], test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize the GradientBoostingClassifier
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the models
rf_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)

# Make predictions with both models on the test set
y_pred_rf = rf_model.predict(X_test)
y_pred_gbm = gbm_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# Evaluate the GBM model
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
classification_rep_gbm = classification_report(y_test, y_pred_gbm)

# Display results using Streamlit
with st.expander("Model Comparison: Random Forest vs GBM"):
    st.write("### Random Forest Results")
    st.write("**Accuracy:**", accuracy_rf)
    st.write("**Classification Report:**")
    st.text(classification_rep_rf)

    st.write("### Gradient Boosting Machine (GBM) Results")
    st.write("**Accuracy:**", accuracy_gbm)
    st.write("**Classification Report:**")
    st.text(classification_rep_gbm)

# Predictions using the Random Forest model on new input data
rf_predictions = rf_model.predict(input_row_scaled)
rf_predictions_prob = rf_model.predict_proba(input_row_scaled)

# Predictions using the GBM model on new input data
gbm_predictions = gbm_model.predict(input_row_scaled)
gbm_predictions_prob = gbm_model.predict_proba(input_row_scaled)


with st.expander('Feature Importance Visualization'):
    # Feature importances from the RandomForest model
    rf_feature_importances = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    # Plot RandomForest feature importances
    fig_rf_importance = px.bar(
        rf_feature_importances,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Random Forest Feature Importance',
        labels={'Importance': 'Importance', 'Feature': 'Feature'},
    )
    
    st.plotly_chart(fig_rf_importance)
    
    # Feature importances from the GradientBoostingClassifier model
    gbm_feature_importances = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': gbm_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    # Plot GradientBoosting feature importances
    fig_gbm_importance = px.bar(
        gbm_feature_importances,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Gradient Boosting Feature Importance',
        labels={'Importance': 'Importance', 'Feature': 'Feature'},
    )
    
    st.plotly_chart(fig_gbm_importance)

# Display predictions in Streamlit
with st.expander("Predictions on New Data"):
    st.write("### Random Forest Predictions")
    st.write(f"Prediction Probabilities: {[f'{prob:.2f}' for prob in rf_predictions_prob[0]]}")

    st.write("### GBM Predictions")
    st.write(f"Prediction Probabilities: {[f'{prob:.2f}' for prob in gbm_predictions_prob[0]]}")



# Map prediction back to the species
predicted_species = uniques[rf_predictions[0]]

# Add a dot to the scatter plot
with st.expander('Data Visualization Prediction'):
    fig = px.scatter(
        df,
        x='bill_length_mm',
        y='body_mass_g',
        color='species',
        title='Penguins Data with Model Prediction'
    )

    # Add the predicted point
    fig.add_scatter(
        x=[bill_length_mm],
        y=[body_mass_g],
        mode='markers',
        marker=dict(size=15, color='white', line=dict(width=2, color=fig.data[list(uniques).index(predicted_species)].marker.color)),
        name=f'Predicted: {predicted_species}'
    )

    st.plotly_chart(fig)


with st.expander('Data Visualization Predictions: 3 the most importent features (static)'):
    # Get the top 3 important features for Random Forest and GBM models
    top_3_rf_features = rf_feature_importances.tail(3)['Feature'].tolist()
    top_3_gbm_features = gbm_feature_importances.tail(3)['Feature'].tolist()
    st.write("### top_3_rf_features")
    st.write(top_3_rf_features)
    st.write("### top_3_gbm_features")
    st.write(top_3_gbm_features)

    st.write("### Random Forest - Feature Combinations")

    # Function to get original features if they exist
    def get_original_feature(feature_name):
        if feature_name in df.columns:
            return feature_name
        elif '_' in feature_name:
            base_feature = feature_name.split('_')[0]
            if base_feature in df.columns:
                return base_feature
        return None

    # Plot combinations of top 3 features for Random Forest
    for i in range(3):
        for j in range(i + 1, 3):
            feature_1 = top_3_rf_features[i]
            feature_2 = top_3_rf_features[j]

            original_feature_1 = get_original_feature(feature_1)
            original_feature_2 = get_original_feature(feature_2)

            if original_feature_1 and original_feature_2:
                fig_rf = px.scatter(
                    df,
                    x=original_feature_1,
                    y=original_feature_2,
                    color='species',
                    title=f'Random Forest Prediction - Features: {original_feature_1} vs {original_feature_2}'
                )

                # Add the predicted point
                fig_rf.add_scatter(
                    x=[input_df[original_feature_1].values[0]],
                    y=[input_df[original_feature_2].values[0]],
                    mode='markers',
                    marker=dict(size=15, color='white', line=dict(width=2, color=fig.data[list(uniques).index(predicted_species)].marker.color)),
                    name=f'Predicted RF: {predicted_species}'
                )

                st.plotly_chart(fig_rf)

    st.write("### Gradient Boosting Machine (GBM) - Feature Combinations")

    # Plot combinations of top 3 features for GBM
    for i in range(3):
        for j in range(i + 1, 3):
            feature_1 = top_3_gbm_features[i]
            feature_2 = top_3_gbm_features[j]

            original_feature_1 = get_original_feature(feature_1)
            original_feature_2 = get_original_feature(feature_2)

            if original_feature_1 and original_feature_2:
                fig_gbm = px.scatter(
                    df,
                    x=original_feature_1,
                    y=original_feature_2,
                    color='species',
                    title=f'GBM Prediction - Features: {original_feature_1} vs {original_feature_2}'
                )

                # Add the predicted point
                fig_gbm.add_scatter(
                    x=[input_df[original_feature_1].values[0]],
                    y=[input_df[original_feature_2].values[0]],
                    mode='markers',
                    marker=dict(size=15, color='white', line=dict(width=2, color=fig.data[list(uniques).index(predicted_species)].marker.color)),
                    name=f'Predicted GBM: {predicted_species}'
                )

                st.plotly_chart(fig_gbm)

with st.expander('Data Visualization Predictions: 3 the most importent features (dynamic)'):
    # Get the top 3 important features for Random Forest and GBM models
    top_3_rf_features = rf_feature_importances.tail(3)['Feature'].tolist()
    top_3_gbm_features = gbm_feature_importances.tail(3)['Feature'].tolist()

    st.write("### Random Forest - Feature Combinations")

    # Initialize checkboxes for selecting features
    selected_rf_features = []
    rf_checkboxes = {}

    for feature in top_3_rf_features:
        rf_checkboxes[feature] = st.checkbox(f"Select {feature}", key=f"rf_{feature}")

    # Ensure only two checkboxes are selected
    for feature, checked in rf_checkboxes.items():
        if checked:
            selected_rf_features.append(feature)

    if len(selected_rf_features) > 2:
        st.error('Please select exactly two features.')
        selected_rf_features = []
    elif len(selected_rf_features) == 2:
        original_feature_1 = get_original_feature(selected_rf_features[0])
        original_feature_2 = get_original_feature(selected_rf_features[1])

        if original_feature_1 and original_feature_2:
            fig_rf = px.scatter(
                df,
                x=original_feature_1,
                y=original_feature_2,
                color='species',
                title=f'Random Forest Prediction - Features: {original_feature_1} vs {original_feature_2}'
            )

            # Add the predicted point
            fig_rf.add_scatter(
                x=[input_df[original_feature_1].values[0]],
                y=[input_df[original_feature_2].values[0]],
                mode='markers',
                marker=dict(size=15, color='white', line=dict(width=2, color=fig_rf.data[list(uniques).index(predicted_species)].marker.color)),
                name=f'Predicted RF: {predicted_species}'
            )

            st.plotly_chart(fig_rf)
        else:
            st.error('Selected features are not available for plotting.')

    st.write("### Gradient Boosting Machine (GBM) - Feature Combinations")

    # Initialize checkboxes for selecting features
    selected_gbm_features = []
    gbm_checkboxes = {}

    for feature in top_3_gbm_features:
        gbm_checkboxes[feature] = st.checkbox(f"Select {feature}", key=f"gbm_{feature}")

    # Ensure only two checkboxes are selected
    for feature, checked in gbm_checkboxes.items():
        if checked:
            selected_gbm_features.append(feature)

    if len(selected_gbm_features) > 2:
        st.error('Please select exactly two features.')
        selected_gbm_features = []
    elif len(selected_gbm_features) == 2:
        original_feature_1 = get_original_feature(selected_gbm_features[0])
        original_feature_2 = get_original_feature(selected_gbm_features[1])

        if original_feature_1 and original_feature_2:
            fig_gbm = px.scatter(
                df,
                x=original_feature_1,
                y=original_feature_2,
                color='species',
                title=f'GBM Prediction - Features: {original_feature_1} vs {original_feature_2}'
            )

            # Add the predicted point
            fig_gbm.add_scatter(
                x=[input_df[original_feature_1].values[0]],
                y=[input_df[original_feature_2].values[0]],
                mode='markers',
                marker=dict(size=15, color='white', line=dict(width=2, color=fig_gbm.data[list(uniques).index(predicted_species)].marker.color)),
                name=f'Predicted GBM: {predicted_species}'
            )

            st.plotly_chart(fig_gbm)
        else:
            st.error('Selected features are not available for plotting.')
