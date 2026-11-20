import copy
import unittest
import numpy as np
from diagnostics.conditions import CONDITIONS, MAIN_CONDITIONS, FEATURE_KEYS, mix_results
from diagnostics.feature_maps import feature_comparison
from diagnostics.io import validate_pair
from diagnostics.reprojection import project, combine, compare
from diagnostics.plot_renderings import crop_box


def outputs(offset):
    result={'depth_preds':[np.full((1,6,8,8),offset+1.,np.float32)]}
    for j,key in enumerate((*FEATURE_KEYS,'match_probs')):
        result[key]=[np.full((6,4,2,2),offset+j+1.,np.float32)]
    return result


def inputs():
    p,h,w,t=6,8,8,2
    def data(n):
        d={'primary_indices':np.arange(p),'primary_rows':np.arange(p)[None],
           'primary_depth':np.ones((1,p,h,w,1,1),np.float32)}
        for group,count in (('context',n),('target',t)):
            d[group+'_image']=np.zeros((1,count,3,h,w),np.float32)
            d[group+'_intrinsics']=np.tile(np.eye(3),(1,count,1,1))
            d[group+'_extrinsics']=np.tile(np.eye(4),(1,count,1,1))
            d[group+'_near']=np.ones((1,count))*.5
            d[group+'_far']=np.ones((1,count))*10
            d[group+'_index']=(np.arange(count)+(100 if group=='target' else 0))[None]
        return d
    a,b=data(p),data(p*3)
    b['source_rows']=np.arange(p,p*3).reshape(1,p,2)
    return a,b


class Conditions(unittest.TestCase):
    def test_all_condition_routes(self):
        a,b=outputs(0),outputs(10)
        for name,spec in CONDITIONS.items():
            with self.subTest(condition=name):
                r=mix_results(a,b,name)
                self.assertIs(r['depth_preds'],(b if spec.depth=='secondary' else a)['depth_preds'])
                self.assertIs(r['match_probs'],(b if spec.confidence=='secondary' else a)['match_probs'])
                for key in FEATURE_KEYS:self.assertIs(r[key],(b if key in spec.feature_keys else a)[key])
                self.assertIsNot(r,a)
        self.assertEqual(a['depth_preds'][0][0,0,0,0],1)
    def test_six_requested_conditions(self):
        self.assertEqual(len(MAIN_CONDITIONS),6)
        for name in ('secondary_features_only','secondary_mv_features_only'):
            self.assertEqual(CONDITIONS[name].depth,'baseline')
            self.assertEqual(CONDITIONS[name].confidence,'baseline')
    def test_shape_mismatch_rejected(self):
        a,b=outputs(0),outputs(10);b['features_mv'][0]=np.zeros((5,4,2,2),np.float32)
        with self.assertRaises(ValueError):mix_results(a,b,'full_secondary')
    def test_intermediate_depth_rejected(self):
        a,b=outputs(0),outputs(10);a['depth_preds']*=2
        with self.assertRaises(ValueError):mix_results(a,b,'baseline')


class Alignment(unittest.TestCase):
    def test_valid(self):validate_pair(*inputs())
    def test_changed_rgb(self):
        a,b=inputs();b['context_image'][0,0,0,0,0]=1
        with self.assertRaises(AssertionError):validate_pair(a,b)
    def test_changed_bounds(self):
        a,b=inputs();b['target_near']*=2
        with self.assertRaises(AssertionError):validate_pair(a,b)
    def test_changed_camera(self):
        a,b=inputs();b['context_extrinsics'][0,0,0,3]=1
        with self.assertRaises(AssertionError):validate_pair(a,b)
    def test_changed_order(self):
        a,b=inputs();b['context_index'][0,:2]=b['context_index'][0,:2][::-1]
        with self.assertRaises(AssertionError):validate_pair(a,b)
    def test_target_leakage(self):
        a,b=inputs();a['target_index'][0,0]=0
        with self.assertRaises(ValueError):validate_pair(a,b)
    def test_overlapping_roles(self):
        a,b=inputs();b['source_rows'][0,0,0]=0
        with self.assertRaises(ValueError):validate_pair(a,b)
    def test_depth_shape(self):
        a,b=inputs();b['primary_depth']=np.ones((1,6,7,8))
        with self.assertRaises(ValueError):validate_pair(a,b)


class Reprojection(unittest.TestCase):
    def test_identity_and_boundaries(self):
        rng=np.random.default_rng(3);rgb=rng.random((8,12,3)).astype(np.float32)
        d=np.ones((8,12));d[0,0]=.5;d[0,1]=10
        warped=project(d,rgb,np.eye(3),np.eye(4),np.eye(3),np.eye(4),.5,10)
        self.assertTrue(np.isfinite(warped[1]).all())
        np.testing.assert_allclose(warped[0],rgb)
    def test_translation_direction(self):
        rgb=np.arange(12,dtype=np.float32).reshape(1,4,3);ta=np.eye(4);ta[0,3]=.25
        warped=project(np.ones((1,4)),rgb,np.eye(3),ta,np.eye(3),np.eye(4),.1,10)
        np.testing.assert_allclose(warped[0][0,1:],rgb[0,:-1])
        self.assertFalse(np.isfinite(warped[1][0,0]))
    def test_nearest_depth_wins_collision(self):
        rgb=np.array([[[1,0,0],[0,1,0]]],np.float32)
        tb=np.eye(4);tb[0,3]=1;kb=np.eye(3);kb[0,2]=.75
        warped=project(np.array([[2.,1.]]),rgb,np.eye(3),np.eye(4),kb,tb,.1,10)
        np.testing.assert_array_equal(warped[0][0,1],[0,1,0])
        self.assertEqual(warped[2][0,1],1)
    def test_behind_camera(self):
        tb=np.diag([-1,1,-1,1])
        warped=project(np.ones((2,2)),np.ones((2,2,3)),np.eye(3),np.eye(4),np.eye(3),tb,.1,10)
        self.assertFalse(np.isfinite(warped[1]).any())
    def test_no_common_coverage_is_undefined(self):
        image=np.zeros((2,2,3));z=np.full((2,2),np.inf)
        scores,_,_,_=compare((image,z),(image,z),image)
        self.assertIsNone(scores['mae_baseline_common'])
    def test_combination_zbuffer(self):
        a=(np.zeros((2,2,3)),np.ones((2,2)))
        b=(np.ones((2,2,3)),np.ones((2,2))*2)
        np.testing.assert_array_equal(combine([a,b])[0],a[0])
    def test_invalid_depth_excluded(self):
        d=np.array([[np.nan,0],[-1,20.]])
        warped=project(d,np.zeros((2,2,3)),np.eye(3),np.eye(4),np.eye(3),np.eye(4),.5,10)
        self.assertFalse(np.isfinite(warped[1]).any())


class FeatureMaps(unittest.TestCase):
    def test_scaling_changes_magnitude_not_direction(self):
        a=np.ones((2,3,4,4),np.float32);maps,_=feature_comparison(a,2*a)
        np.testing.assert_allclose(maps['cosine'],1)
        np.testing.assert_allclose(maps['relative_difference'],1)
    def test_zero_vectors_marked_undefined(self):
        maps,rows=feature_comparison(np.zeros((1,2,2,2)),np.ones((1,2,2,2)))
        self.assertTrue(np.isnan(maps['cosine']).all())
        self.assertIsNone(rows[0]['mean_spatial_cosine'])
    def test_crop_bounds(self):
        self.assertEqual(crop_box(None,8,12),(3,2,6,4))
        with self.assertRaises(ValueError):crop_box((0,0,20,20),8,12)


if __name__=='__main__':unittest.main()
