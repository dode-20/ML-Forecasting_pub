import streamlit as st

st.title("Documentation")

st.markdown("This page displays some relevant charts and diagrams used throughout the project. The following are embedded Miro boards or placeholder visualizations:")

st.markdown("## General project")
st.markdown("### Project Structure")
st.components.v1.html(
    '''
    <iframe width="100%" height="432" style="display:block; max-width:100%;" src="https://miro.com/app/live-embed/uXjVINWuZhc=/?focusWidget=3458764621958268681&embedMode=view_only_without_ui&embedId=43289080768" frameborder="0" scrolling="no" allow="fullscreen; clipboard-read; clipboard-write" allowfullscreen></iframe>
    ''',
    height=450,
)

st.markdown("### File Structure")
st.components.v1.html(
    '''
    <iframe width="100%" height="432" style="display:block; max-width:100%;" src="https://miro.com/app/live-embed/uXjVINWuZhc=/?focusWidget=3458764625583663532&embedMode=view_only_without_ui&embedId=914297190797" frameborder="0" scrolling="no" allow="fullscreen; clipboard-read; clipboard-write" allowfullscreen></iframe>
    ''',
    height=450,
)

st.markdown("---")
st.markdown("## Data Processing")
st.markdown("### InfluxDB - getting the data into the py-script")
st.components.v1.html(
    '''
    <iframe width="100%" height="432" style="display:block; max-width:100%;" src="https://miro.com/app/live-embed/uXjVINWuZhc=/?focusWidget=3458764627359920064&embedMode=view_only_without_ui&embedId=810011725610" frameborder="0" scrolling="no" allow="fullscreen; clipboard-read; clipboard-write" allowfullscreen></iframe>
    ''',
    height=450,
)
