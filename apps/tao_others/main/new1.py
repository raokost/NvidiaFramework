import sys
sys.path.append('../')
import platform
import configparser
from dataclasses import dataclass
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst, GObject
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import ctypes
import numpy as np
import math
import pyds
# from common import gst, unittest, TestCase, pygobject_2_13

fps_streams={}
MAX_DISPLAY_LEN=64
MEASURE_ENABLE =1
PGIE_CLASS_ID_FACE =0
PGIE_DETECTED_CLASS_NUM= 4
MUXER_OUTPUT_WIDTH =1280
MUXER_OUTPUT_HEIGHT= 720
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
MUXER_BATCH_TIMEOUT_USEC =4000000
GST_CAPS_FEATURES_NVMM= "memory:NVMM"
CONFIG_GPU_ID= "gpu-id"
SGIE_NET_WIDTH =80
SGIE_NET_HEIGHT= 80
total_face_num = 0
frame_number = 1
PRIMARY_DETECTOR_UID= 1
SECOND_DETECTOR_UID =2
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1






def osd_sink_pad_buffer_probe(pad, info, u_data):
    global total_face_num
    global frame_number
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    face_count = 0
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                if obj_meta.class_id == PGIE_CLASS_ID_FACE:
                    face_count += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
            
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    
    print("Frame Number =", frame_number,  "Face Count =", face_count)    
    total_face_num += face_count
    frame_number += 1
    return Gst.PadProbeReturn.OK


def tile_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    part_index = 0
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            facebboxdraw = False
            left_gaze_x1 = 0
            left_gaze_y1 = 0
            left_gaze_x2 = 0
            left_gaze_y2 = 0
            right_gaze_x1 = 0
            right_gaze_y1 = 0
            right_gaze_x2 = 0
            right_gaze_y2 = 0
            
            l_user = obj_meta.obj_user_meta_list
            
            while l_user is not None:
                user_meta=pyds.NvDsUserMeta.cast(l_user.data)
                gaze = pyds.NvDsGazeMetaData.cast(user_meta.user_meta_data)
                print(dir(gaze))
                try: 
                    l_user=l_user.next
                except StopIteration:
                    break
      
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
            
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        
    return Gst.PadProbeReturn.OK






def sgie_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            l_user = obj_meta.obj_user_meta_list
            
            while l_user is not None:
                user_meta=pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    continue
                
                
                confidence = None
                heatmap_data = None
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    l_name = layer.layerName
                    if l_name == "softargmax":
                        heatmap_data = tensor_meta.out_buf_ptrs_host
                    elif l_name == "softargmax:1":
                        confidence = tensor_meta.out_buf_ptrs_host
                        
                print("check2")
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                v = np.ctypeslib.as_array(ptr, shape=(5,))
                print(v)
                
                    
                # print("check3", confidence, heatmap_data, "jokl")
                    
                try: 
                    l_user=l_user.next
                except StopIteration:
                    break
                
                
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
            
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        
    return Gst.PadProbeReturn.OK


    
    
    
def cb_newpad1(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)


    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")
            
            
def decodebin_child_added1(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added1,user_data) 


def create_source_bin1(index,uri):
    print("Creating source bin")
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added",cb_newpad1,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added1,nbin)

    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin
    
    
    
def split(s, p):
    return s.split(p)


def main(args):
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    for i in range(0,len(args)-1):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
    number_sources=len(args)-1

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin1(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")

    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
    
    
    

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    
    print("Creating Sgie \n ")
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    if not sgie:
        sys.stderr.write(" Unable to create sgie \n")
        
    print("Creating Emotioninfer \n ")
    emotioninfer = Gst.ElementFactory.make("nvdsvideotemplate", "emotioninfer")
    if not emotioninfer:
        sys.stderr.write(" Unable to create Emotioninfer \n")
        
    print("Creating Gaze_identifier \n ")
    gaze_identifier = Gst.ElementFactory.make("nvdsvideotemplate", "gaze_identifier")
    if not gaze_identifier:
        sys.stderr.write(" Unable to create Gaze_identifier \n")
    
    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
        
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
        
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
        
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    

    print("Creating Fakesink \n")
    sink = Gst.ElementFactory.make("fakesink", "fake-renderer")
    if not sink:
        sys.stderr.write(" Unable to create fake sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 40000)
    
    
    pgie.set_property('config-file-path', "../../../configs/facial_tao/config_infer_primary_facenet.txt")
    pgie.set_property("unique-id", PRIMARY_DETECTOR_UID)
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)


    sgie.set_property('config-file-path', "../../../configs/facial_tao/faciallandmark_sgie_config.txt")
    sgie.set_property("unique-id", SECOND_DETECTOR_UID)
    sgie_batch_size=sgie.get_property("batch-size")
    if(sgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",sgie_batch_size," with number of sources ", number_sources," \n")
        sgie.set_property("batch-size",number_sources)
        
    emotioninfer.set_property("customlib-name", "./emotion_impl/libnvds_emotion_impl.so")
    emotioninfer.set_property("customlib-props", "config-file:../../../configs/emotion_tao/sample_emotion_model_config.txt")
    
    gaze_identifier.set_property("customlib-name", "./gazeinfer_impl/libnvds_gazeinfer.so")
    gaze_identifier.set_property("customlib-props", "config-file:../../../configs/gaze_tao/sample_gazenet_model_config.txt")
    

    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)
    

    
    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(emotioninfer)
    pipeline.add(gaze_identifier)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(sgie)
    sgie.link(queue3)
    queue3.link(gaze_identifier)
    gaze_identifier.link(queue4)
    queue4.link(emotioninfer)
    emotioninfer.link(queue5)
    queue5.link(tiler)
    tiler.link(queue6)
    queue6.link(nvvidconv)
    nvvidconv.link(queue7)
    queue7.link(nvosd)
    nvosd.link(queue8)
    queue8.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    osd_sink_pad = nvosd.get_static_pad ("sink")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get sink pad\n");
    else:
         osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tile_sink_pad_buffer_probe, 0)
     
    print("hello")   
    osd_sink_pad = nvosd.get_static_pad ("sink")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get sink pad\n");
    else:
         osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
         
                  
         
    osd_sink_pad = queue3.get_static_pad ("src")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get src pad\n");
    else:
         osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    print("Totally ", total_face_num, " faces are inferred\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
