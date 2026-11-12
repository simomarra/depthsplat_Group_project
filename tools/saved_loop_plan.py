"""Map the inspected loop selection to converted rows using filename IDs."""
import re
import numpy as np


def load_saved_loop_plan(saved, timestamps, scene, num_primary=6, num_targets=8):
    if saved.get('scene_id') != scene:
        raise ValueError('Saved plan belongs to a different scene.')
    raw_ids = saved['source_frame_ids_chronological']
    converted = np.asarray(timestamps)
    if converted.ndim != 1 or not np.issubdtype(converted.dtype, np.integer):
        raise ValueError('Converted timestamps must be a 1D integer array.')
    converted = converted.tolist()
    if len(set(converted)) != len(converted):
        raise ValueError('Duplicate converted filename IDs.')
    if (len(raw_ids) != saved['total_frames'] or len(set(raw_ids)) != len(raw_ids)
            or raw_ids != sorted(raw_ids) or set(raw_ids) != set(converted)):
        raise ValueError('Converted scene frame IDs differ from the verified chronological catalog.')
    mapping = {frame_id: row for row, frame_id in enumerate(converted)}
    end = saved['loop']['end_row']
    if saved['loop']['start_row'] != 0 or saved['eligible_range'] != [0,end-1] or not 1 < end < len(raw_ids):
        raise ValueError('Unexpected loop bounds.')

    def file_id(name):
        match = re.fullmatch(r'frame_(\d+)\.png', name)
        if match is None: raise ValueError(f'Unexpected filename {name}.')
        return int(match.group(1))

    if file_id(saved['closing_filename']) != raw_ids[end]:
        raise ValueError('Closing filename and row disagree.')

    def map_rows(rows, filenames=None):
        if not isinstance(rows,list) or any(type(i) is not int or i < 0 or i >= end for i in rows):
            raise ValueError('Selected rows must be integer rows inside the saved loop.')
        ids = [raw_ids[i] for i in rows]
        if filenames is not None and [file_id(name) for name in filenames] != ids:
            raise ValueError('Saved row indices disagree with filenames.')
        return [mapping[i] for i in ids]

    targets = map_rows(saved['target_rows'], saved['target_files'])
    if len(targets) != num_targets or len(set(targets)) != num_targets:
        raise ValueError('Wrong target count or repeated targets.')
    methods = {}
    labels = {'fps':'Original-rule FPS within loop','trajectory':'Uniform distance within loop'}
    for name,label in labels.items():
        method = saved['methods'][label]
        primary = map_rows(method['primary_rows'],method['primary_files'])
        if len(primary)!=num_primary or len(method['secondary_rows'])!=num_primary or len(method['secondary_files'])!=num_primary:
            raise ValueError('Wrong primary/support row count.')
        supports=[]
        for rows,files in zip(method['secondary_rows'],method['secondary_files']):
            if len(rows)!=2 or len(files)!=2: raise ValueError('Require two supports per primary.')
            supports.append(map_rows(rows,files))
        all_inputs=primary+sum(supports,[])
        if len(set(all_inputs))!=3*num_primary or set(all_inputs)&set(targets):
            raise ValueError('Inputs must be unique and disjoint from held-out targets.')
        methods[name]={'primary':primary,'secondary':supports,
                       'primary_files':method['primary_files'],'secondary_files':method['secondary_files']}
    return {'frame_count':len(converted),'num_primary':num_primary,'num_secondary':2*num_primary,
            'target_indices':targets,'target_files':saved['target_files'],
            'support_offset':saved['support_offset'],'methods':methods,
            'loop':saved['loop'],'raw_eligible_range':saved['eligible_range'],
            'eligible_converted_rows':[mapping[i] for i in raw_ids[:end]],
            'converted_source_frame_ids':converted,
            'target_rule':'exact saved held-out targets; no resampling',
            'fps':'exact inspected NumPy original-rule FPS selections; not rerun in Torch',
            'frame_order':'mapped by filename ID; dataset row indices are recorded separately'}
