# ideally specify working directory first below as: base_dir = 'path/to/working/directory'
# (but can also choose from sidebar later)
# run after activating environment with $ conda activate stlit-env
# using the command $ streamlit run image_labeler.py

# specify the image directory (change this)
base_dir = '/Users/parndt/jupyterprojects/GLD3_complete/AIS' 

import streamlit as st
import os
import numpy as np
import shutil
from PIL import Image
from streamlit_shortcuts import add_keyboard_shortcuts

# change padding to make app more compact
st.markdown("""
    <style>
           .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 0rem;
                padding-right: 0rem;
            }
    </style>
    """, unsafe_allow_html=True)

image_dir = base_dir + '/detection_context'
st.sidebar.header("Working Directory")
image_directory = st.sidebar.text_input("Image Directory", image_dir)

# make space for displaying the current image
last_action_confirm = st.empty()
current_image = st.empty()

# create columns with action buttons, and add keyboard shortcuts
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col2:
    good_button = st.button(label=":green[**g**: good]", key='good_lake', use_container_width=True)
with col3:
    bad_button = st.button(label=":red[**b**: no lake]", key='bad_no_lake', use_container_width=True)
with col4:
    unsure_button = st.button(label=":blue[**u**: bad depth]", key='lake_bad_depth', use_container_width=True)
with col5:
    interesting_button = st.button(label=":gray[**i**: interesting]", key='interesting', use_container_width=True)
add_keyboard_shortcuts({'g': 'good_lake', 'b': 'bad_no_lake', 'u': 'lake_bad_depth', 'i': 'interesting'})

# add columns for info section in the bottom
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    counts_totals = st.empty()
with c2: 
    counts_labeled = st.empty()
with c4:
    last_label = st.empty()
with c5: 
    last_image = st.empty()
image_index = st.empty()

# main function
def main():
    
    # Initialization
    if 'n_img' not in st.session_state:
        st.session_state['n_img'] = 0
    if 'last_img' not in st.session_state:
        st.session_state['last_img'] = 'none'
    if 'image_files' not in st.session_state:
        img_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        img_files.sort()
        st.session_state['image_files'] = img_files
    if 'counts' not in st.session_state:
        foldernames = ['detection_context/' + x for x in ['good_lake', 'bad_no_lake', 'lake_bad_depth', 'interesting']]
        folderpaths = [image_dir[:image_dir.rfind('/')] + '/' + x for x in foldernames]
        for path in folderpaths:
            if not os.path.exists(path):
                os.makedirs(path)
        counts_init = [len([f for f in os.listdir(fp) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) for fp in folderpaths]
        st.session_state['counts'] = dict(zip(foldernames, counts_init))
        st.session_state['count_unsorted'] = len(st.session_state.image_files)
        st.session_state['count_labeled'] = np.sum(list(st.session_state.counts.values())[:3])
        st.session_state['count_total'] = st.session_state.count_unsorted + st.session_state.count_labeled
    
    # update valued based on last button
    label = None
    txtcol = 'gray'
    if good_button:
        label = 'good_lake'
        txtcol = 'green'
        st.session_state['count_unsorted'] -= 1
        st.session_state['count_labeled'] += 1
        st.session_state['n_img'] += 1
    elif bad_button:
        label = 'bad_no_lake'
        txtcol = 'red'
        st.session_state['count_unsorted'] -= 1
        st.session_state['count_labeled'] += 1
        st.session_state['n_img'] += 1
    elif unsure_button:
        label = 'lake_bad_depth'
        txtcol = 'blue'
        st.session_state['count_unsorted'] -= 1
        st.session_state['count_labeled'] += 1
        st.session_state['n_img'] += 1
    elif interesting_button:
        label = 'interesting'
        txtcol = 'gray'

    if label:
        st.session_state['counts']['detection_context/' + label] += 1

    i = st.session_state['n_img']

    # show the current image
    if i <= len(st.session_state.image_files):
        image_file = st.session_state.image_files[i]
        image_path = os.path.join(image_directory, image_file)
        if label: 
            last_action_confirm.markdown(f":{txtcol}[**--> Last image was sucessfully labeled as {label}!**]")
        current_image.image(Image.open(image_path), use_column_width=True)
    else:
        current_image.text('DONE!')
    image_index.text(f"image {st.session_state.count_labeled}/{len(st.session_state.image_files)}: {image_file}")

    # write progress info summary
    pct_lab = int(np.round(st.session_state.count_labeled / st.session_state.count_total * 100))
    counts_totals.markdown(f'- {st.session_state.count_total} total \n- {st.session_state.count_unsorted} remaining \n- {pct_lab}% labeled')
    labeldict = {'detection_context/good_lake': 'good', 
                 'detection_context/bad_no_lake': 'no lake', 
                 'detection_context/lake_bad_depth': 'bad depth', 
                 'detection_context/interesting': 'interesting'}
    counts_labeled.markdown(''.join([f'- {v} {labeldict[k]}\n' for k, v in st.session_state.counts.items()]))

    # do the file moving at the end
    if label == 'interesting':
        if i < len(st.session_state.image_files):
            imagetocopy_src = os.path.join(image_directory, st.session_state.image_files[i])
            imagetocopy_dst = base_dir + '/detection_context/' + label
            shutil.copy2(imagetocopy_src, imagetocopy_dst)
            st.session_state['last_img'] = imagetocopy_dst + '/' + st.session_state.image_files[i]
    elif label in ['good_lake', 'bad_no_lake', 'lake_bad_depth']:
        if i < len(st.session_state.image_files):
            imagetomove_src = os.path.join(image_directory, st.session_state.image_files[i-1])
            imagetomove_dst = base_dir + '/detection_context/' + label
            shutil.move(imagetomove_src, imagetomove_dst)
            st.session_state['last_img'] = imagetomove_dst + '/' + st.session_state.image_files[i-1]

    # show the last image action (if applicable)
    if st.session_state['last_img'] != 'none':
        if label=='interesting':
            last_label.write(f':{txtcol}[This plot labeled as **{label}**:]')
            last_image.image(Image.open(st.session_state['last_img']), 
                             caption=f"Image {i+1}", use_column_width=True)
        else:
            last_label.write(f':{txtcol}[Last plot labeled as **{label}**:]')
            last_image.image(Image.open(st.session_state['last_img']), 
                             caption=f"Image {i}", use_column_width=True)

    # stop at end
    if i >= len(st.session_state.image_files):
        st.warning('Done! Stopping the app.')
        st.stop()

if __name__ == "__main__":
    main()
