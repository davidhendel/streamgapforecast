import pal5_util
import gd1_util
import phx_util
from galpy.util import save_pickles
from streampepper_utils import parse_times
from multiprocessing import Pool

'''
Setup the models used to calculate mock streams.
64 time samplings is expensive but makes a big difference
New streams can be added with new _util files; key changes needed are
 1. Orbit, age, sigv to reproduce your desired streams
 2. Choose an appropriate b for actionAngleIsochroneApprox,
	can use galpy.actionAngle.estimateBIsochrone

To sample the stream density you will need
 1. A method to return stars in the desired coordinate system, 
 	e.g. pal5_util.radec_to_pal5xieta() or phx_util.galactic_to_phoenix()
 2. Some choice of observed area

'''

#pal5_leading_model  = pal5_util.setup_pal5model(timpact=parse_times('1sampling',5.0),leading=True)
#pal5_trailing_model = pal5_util.setup_pal5model(timpact=parse_times('1sampling',5.0),leading=False)
#gd1_leading_model   = gd1_util.setup_gd1model(  timpact=parse_times('1sampling',9.0),leading=True)
#gd1_trailing_model  = gd1_util.setup_gd1model(  timpact=parse_times('1sampling',9.0),leading=False)
#phx_leading_model   = phx_util.setup_phxmodel(  timpact=parse_times('1sampling',1.5),leading=True)
#phx_trailing_model  = phx_util.setup_phxmodel(  timpact=parse_times('1sampling',1.5),leading=False)

args = [
('pal5_leading_2sampling.pkl' ,pal5_util.setup_pal5model, parse_times('2sampling',5.0), True ),
('pal5_trailing_2sampling.pkl',pal5_util.setup_pal5model, parse_times('2sampling',5.0), False),
('gd1_leading_2sampling.pkl'  ,gd1_util.setup_gd1model  , parse_times('2sampling',9.0), True ),
('gd1_trailing_2sampling.pkl' ,gd1_util.setup_gd1model  , parse_times('2sampling',9.0), False),
('phx_leading_2sampling.pkl'  ,phx_util.setup_phxmodel  , parse_times('2sampling',1.5), True ),
('phx_trailing_2sampling.pkl' ,phx_util.setup_phxmodel  , parse_times('2sampling',1.5), False),
]

def save_stream_model_pickles(fname, setupfunc, timpact, leading):
	model = setupfunc(timpact=timpact, leading=leading)
	save_pickles(fname, model)
	return

if __name__ == '__main__':
    p = Pool(6)
    print(p.map(save_stream_model_pickles, args))













