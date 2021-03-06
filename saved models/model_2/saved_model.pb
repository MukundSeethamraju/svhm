??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
&simple_model2/conv_block/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&simple_model2/conv_block/conv2d/kernel
?
:simple_model2/conv_block/conv2d/kernel/Read/ReadVariableOpReadVariableOp&simple_model2/conv_block/conv2d/kernel*&
_output_shapes
: *
dtype0
?
$simple_model2/conv_block/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$simple_model2/conv_block/conv2d/bias
?
8simple_model2/conv_block/conv2d/bias/Read/ReadVariableOpReadVariableOp$simple_model2/conv_block/conv2d/bias*
_output_shapes
: *
dtype0
?
(simple_model2/conv_block/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *9
shared_name*(simple_model2/conv_block/conv2d_1/kernel
?
<simple_model2/conv_block/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp(simple_model2/conv_block/conv2d_1/kernel*&
_output_shapes
:  *
dtype0
?
&simple_model2/conv_block/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&simple_model2/conv_block/conv2d_1/bias
?
:simple_model2/conv_block/conv2d_1/bias/Read/ReadVariableOpReadVariableOp&simple_model2/conv_block/conv2d_1/bias*
_output_shapes
: *
dtype0
?
*simple_model2/conv_block_1/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*;
shared_name,*simple_model2/conv_block_1/conv2d_2/kernel
?
>simple_model2/conv_block_1/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp*simple_model2/conv_block_1/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
?
(simple_model2/conv_block_1/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(simple_model2/conv_block_1/conv2d_2/bias
?
<simple_model2/conv_block_1/conv2d_2/bias/Read/ReadVariableOpReadVariableOp(simple_model2/conv_block_1/conv2d_2/bias*
_output_shapes
:@*
dtype0
?
*simple_model2/conv_block_1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*;
shared_name,*simple_model2/conv_block_1/conv2d_3/kernel
?
>simple_model2/conv_block_1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp*simple_model2/conv_block_1/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
?
(simple_model2/conv_block_1/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(simple_model2/conv_block_1/conv2d_3/bias
?
<simple_model2/conv_block_1/conv2d_3/bias/Read/ReadVariableOpReadVariableOp(simple_model2/conv_block_1/conv2d_3/bias*
_output_shapes
:@*
dtype0
?
/simple_model2/larger_conv_block/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*@
shared_name1/simple_model2/larger_conv_block/conv2d_4/kernel
?
Csimple_model2/larger_conv_block/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp/simple_model2/larger_conv_block/conv2d_4/kernel*'
_output_shapes
:@?*
dtype0
?
-simple_model2/larger_conv_block/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-simple_model2/larger_conv_block/conv2d_4/bias
?
Asimple_model2/larger_conv_block/conv2d_4/bias/Read/ReadVariableOpReadVariableOp-simple_model2/larger_conv_block/conv2d_4/bias*
_output_shapes	
:?*
dtype0
?
/simple_model2/larger_conv_block/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*@
shared_name1/simple_model2/larger_conv_block/conv2d_5/kernel
?
Csimple_model2/larger_conv_block/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp/simple_model2/larger_conv_block/conv2d_5/kernel*(
_output_shapes
:??*
dtype0
?
-simple_model2/larger_conv_block/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-simple_model2/larger_conv_block/conv2d_5/bias
?
Asimple_model2/larger_conv_block/conv2d_5/bias/Read/ReadVariableOpReadVariableOp-simple_model2/larger_conv_block/conv2d_5/bias*
_output_shapes	
:?*
dtype0
?
/simple_model2/larger_conv_block/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*@
shared_name1/simple_model2/larger_conv_block/conv2d_6/kernel
?
Csimple_model2/larger_conv_block/conv2d_6/kernel/Read/ReadVariableOpReadVariableOp/simple_model2/larger_conv_block/conv2d_6/kernel*(
_output_shapes
:??*
dtype0
?
-simple_model2/larger_conv_block/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-simple_model2/larger_conv_block/conv2d_6/bias
?
Asimple_model2/larger_conv_block/conv2d_6/bias/Read/ReadVariableOpReadVariableOp-simple_model2/larger_conv_block/conv2d_6/bias*
_output_shapes	
:?*
dtype0
?
,simple_model2/prediction_block2/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,simple_model2/prediction_block2/dense/kernel
?
@simple_model2/prediction_block2/dense/kernel/Read/ReadVariableOpReadVariableOp,simple_model2/prediction_block2/dense/kernel* 
_output_shapes
:
??*
dtype0
?
*simple_model2/prediction_block2/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*simple_model2/prediction_block2/dense/bias
?
>simple_model2/prediction_block2/dense/bias/Read/ReadVariableOpReadVariableOp*simple_model2/prediction_block2/dense/bias*
_output_shapes	
:?*
dtype0
?
.simple_model2/prediction_block2/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*?
shared_name0.simple_model2/prediction_block2/dense_1/kernel
?
Bsimple_model2/prediction_block2/dense_1/kernel/Read/ReadVariableOpReadVariableOp.simple_model2/prediction_block2/dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
,simple_model2/prediction_block2/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,simple_model2/prediction_block2/dense_1/bias
?
@simple_model2/prediction_block2/dense_1/bias/Read/ReadVariableOpReadVariableOp,simple_model2/prediction_block2/dense_1/bias*
_output_shapes	
:?*
dtype0
?
.simple_model2/prediction_block2/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.simple_model2/prediction_block2/dense_2/kernel
?
Bsimple_model2/prediction_block2/dense_2/kernel/Read/ReadVariableOpReadVariableOp.simple_model2/prediction_block2/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
,simple_model2/prediction_block2/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,simple_model2/prediction_block2/dense_2/bias
?
@simple_model2/prediction_block2/dense_2/bias/Read/ReadVariableOpReadVariableOp,simple_model2/prediction_block2/dense_2/bias*
_output_shapes
:*
dtype0
?
.simple_model2/prediction_block2/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.simple_model2/prediction_block2/dense_3/kernel
?
Bsimple_model2/prediction_block2/dense_3/kernel/Read/ReadVariableOpReadVariableOp.simple_model2/prediction_block2/dense_3/kernel*
_output_shapes
:	?*
dtype0
?
,simple_model2/prediction_block2/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,simple_model2/prediction_block2/dense_3/bias
?
@simple_model2/prediction_block2/dense_3/bias/Read/ReadVariableOpReadVariableOp,simple_model2/prediction_block2/dense_3/bias*
_output_shapes
:*
dtype0
?
.simple_model2/prediction_block2/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.simple_model2/prediction_block2/dense_4/kernel
?
Bsimple_model2/prediction_block2/dense_4/kernel/Read/ReadVariableOpReadVariableOp.simple_model2/prediction_block2/dense_4/kernel*
_output_shapes
:	?*
dtype0
?
,simple_model2/prediction_block2/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,simple_model2/prediction_block2/dense_4/bias
?
@simple_model2/prediction_block2/dense_4/bias/Read/ReadVariableOpReadVariableOp,simple_model2/prediction_block2/dense_4/bias*
_output_shapes
:*
dtype0
?
.simple_model2/prediction_block2/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.simple_model2/prediction_block2/dense_5/kernel
?
Bsimple_model2/prediction_block2/dense_5/kernel/Read/ReadVariableOpReadVariableOp.simple_model2/prediction_block2/dense_5/kernel*
_output_shapes
:	?*
dtype0
?
,simple_model2/prediction_block2/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,simple_model2/prediction_block2/dense_5/bias
?
@simple_model2/prediction_block2/dense_5/bias/Read/ReadVariableOpReadVariableOp,simple_model2/prediction_block2/dense_5/bias*
_output_shapes
:*
dtype0
?
.simple_model2/prediction_block2/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.simple_model2/prediction_block2/dense_6/kernel
?
Bsimple_model2/prediction_block2/dense_6/kernel/Read/ReadVariableOpReadVariableOp.simple_model2/prediction_block2/dense_6/kernel*
_output_shapes
:	?*
dtype0
?
,simple_model2/prediction_block2/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,simple_model2/prediction_block2/dense_6/bias
?
@simple_model2/prediction_block2/dense_6/bias/Read/ReadVariableOpReadVariableOp,simple_model2/prediction_block2/dense_6/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
?
-Adam/simple_model2/conv_block/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/simple_model2/conv_block/conv2d/kernel/m
?
AAdam/simple_model2/conv_block/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/simple_model2/conv_block/conv2d/kernel/m*&
_output_shapes
: *
dtype0
?
+Adam/simple_model2/conv_block/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/simple_model2/conv_block/conv2d/bias/m
?
?Adam/simple_model2/conv_block/conv2d/bias/m/Read/ReadVariableOpReadVariableOp+Adam/simple_model2/conv_block/conv2d/bias/m*
_output_shapes
: *
dtype0
?
/Adam/simple_model2/conv_block/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *@
shared_name1/Adam/simple_model2/conv_block/conv2d_1/kernel/m
?
CAdam/simple_model2/conv_block/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/simple_model2/conv_block/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
?
-Adam/simple_model2/conv_block/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/simple_model2/conv_block/conv2d_1/bias/m
?
AAdam/simple_model2/conv_block/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp-Adam/simple_model2/conv_block/conv2d_1/bias/m*
_output_shapes
: *
dtype0
?
1Adam/simple_model2/conv_block_1/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*B
shared_name31Adam/simple_model2/conv_block_1/conv2d_2/kernel/m
?
EAdam/simple_model2/conv_block_1/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/simple_model2/conv_block_1/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
?
/Adam/simple_model2/conv_block_1/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/simple_model2/conv_block_1/conv2d_2/bias/m
?
CAdam/simple_model2/conv_block_1/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp/Adam/simple_model2/conv_block_1/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
?
1Adam/simple_model2/conv_block_1/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*B
shared_name31Adam/simple_model2/conv_block_1/conv2d_3/kernel/m
?
EAdam/simple_model2/conv_block_1/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp1Adam/simple_model2/conv_block_1/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
?
/Adam/simple_model2/conv_block_1/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/simple_model2/conv_block_1/conv2d_3/bias/m
?
CAdam/simple_model2/conv_block_1/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp/Adam/simple_model2/conv_block_1/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
?
6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*G
shared_name86Adam/simple_model2/larger_conv_block/conv2d_4/kernel/m
?
JAdam/simple_model2/larger_conv_block/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/m*'
_output_shapes
:@?*
dtype0
?
4Adam/simple_model2/larger_conv_block/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64Adam/simple_model2/larger_conv_block/conv2d_4/bias/m
?
HAdam/simple_model2/larger_conv_block/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOp4Adam/simple_model2/larger_conv_block/conv2d_4/bias/m*
_output_shapes	
:?*
dtype0
?
6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*G
shared_name86Adam/simple_model2/larger_conv_block/conv2d_5/kernel/m
?
JAdam/simple_model2/larger_conv_block/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/m*(
_output_shapes
:??*
dtype0
?
4Adam/simple_model2/larger_conv_block/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64Adam/simple_model2/larger_conv_block/conv2d_5/bias/m
?
HAdam/simple_model2/larger_conv_block/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOp4Adam/simple_model2/larger_conv_block/conv2d_5/bias/m*
_output_shapes	
:?*
dtype0
?
6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*G
shared_name86Adam/simple_model2/larger_conv_block/conv2d_6/kernel/m
?
JAdam/simple_model2/larger_conv_block/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/m*(
_output_shapes
:??*
dtype0
?
4Adam/simple_model2/larger_conv_block/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64Adam/simple_model2/larger_conv_block/conv2d_6/bias/m
?
HAdam/simple_model2/larger_conv_block/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOp4Adam/simple_model2/larger_conv_block/conv2d_6/bias/m*
_output_shapes	
:?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*D
shared_name53Adam/simple_model2/prediction_block2/dense/kernel/m
?
GAdam/simple_model2/prediction_block2/dense/kernel/m/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
1Adam/simple_model2/prediction_block2/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31Adam/simple_model2/prediction_block2/dense/bias/m
?
EAdam/simple_model2/prediction_block2/dense/bias/m/Read/ReadVariableOpReadVariableOp1Adam/simple_model2/prediction_block2/dense/bias/m*
_output_shapes	
:?*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*F
shared_name75Adam/simple_model2/prediction_block2/dense_1/kernel/m
?
IAdam/simple_model2/prediction_block2/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53Adam/simple_model2/prediction_block2/dense_1/bias/m
?
GAdam/simple_model2/prediction_block2/dense_1/bias/m/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_2/kernel/m
?
IAdam/simple_model2/prediction_block2/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_2/bias/m
?
GAdam/simple_model2/prediction_block2/dense_2/bias/m/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_2/bias/m*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_3/kernel/m
?
IAdam/simple_model2/prediction_block2/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_3/bias/m
?
GAdam/simple_model2/prediction_block2/dense_3/bias/m/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_3/bias/m*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_4/kernel/m
?
IAdam/simple_model2/prediction_block2/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_4/kernel/m*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_4/bias/m
?
GAdam/simple_model2/prediction_block2/dense_4/bias/m/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_4/bias/m*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_5/kernel/m
?
IAdam/simple_model2/prediction_block2/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_5/kernel/m*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_5/bias/m
?
GAdam/simple_model2/prediction_block2/dense_5/bias/m/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_5/bias/m*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_6/kernel/m
?
IAdam/simple_model2/prediction_block2/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_6/kernel/m*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_6/bias/m
?
GAdam/simple_model2/prediction_block2/dense_6/bias/m/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_6/bias/m*
_output_shapes
:*
dtype0
?
-Adam/simple_model2/conv_block/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/simple_model2/conv_block/conv2d/kernel/v
?
AAdam/simple_model2/conv_block/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/simple_model2/conv_block/conv2d/kernel/v*&
_output_shapes
: *
dtype0
?
+Adam/simple_model2/conv_block/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/simple_model2/conv_block/conv2d/bias/v
?
?Adam/simple_model2/conv_block/conv2d/bias/v/Read/ReadVariableOpReadVariableOp+Adam/simple_model2/conv_block/conv2d/bias/v*
_output_shapes
: *
dtype0
?
/Adam/simple_model2/conv_block/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *@
shared_name1/Adam/simple_model2/conv_block/conv2d_1/kernel/v
?
CAdam/simple_model2/conv_block/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/simple_model2/conv_block/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
?
-Adam/simple_model2/conv_block/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/simple_model2/conv_block/conv2d_1/bias/v
?
AAdam/simple_model2/conv_block/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp-Adam/simple_model2/conv_block/conv2d_1/bias/v*
_output_shapes
: *
dtype0
?
1Adam/simple_model2/conv_block_1/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*B
shared_name31Adam/simple_model2/conv_block_1/conv2d_2/kernel/v
?
EAdam/simple_model2/conv_block_1/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/simple_model2/conv_block_1/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
?
/Adam/simple_model2/conv_block_1/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/simple_model2/conv_block_1/conv2d_2/bias/v
?
CAdam/simple_model2/conv_block_1/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp/Adam/simple_model2/conv_block_1/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
?
1Adam/simple_model2/conv_block_1/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*B
shared_name31Adam/simple_model2/conv_block_1/conv2d_3/kernel/v
?
EAdam/simple_model2/conv_block_1/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp1Adam/simple_model2/conv_block_1/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
?
/Adam/simple_model2/conv_block_1/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/simple_model2/conv_block_1/conv2d_3/bias/v
?
CAdam/simple_model2/conv_block_1/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp/Adam/simple_model2/conv_block_1/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
?
6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*G
shared_name86Adam/simple_model2/larger_conv_block/conv2d_4/kernel/v
?
JAdam/simple_model2/larger_conv_block/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/v*'
_output_shapes
:@?*
dtype0
?
4Adam/simple_model2/larger_conv_block/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64Adam/simple_model2/larger_conv_block/conv2d_4/bias/v
?
HAdam/simple_model2/larger_conv_block/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOp4Adam/simple_model2/larger_conv_block/conv2d_4/bias/v*
_output_shapes	
:?*
dtype0
?
6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*G
shared_name86Adam/simple_model2/larger_conv_block/conv2d_5/kernel/v
?
JAdam/simple_model2/larger_conv_block/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/v*(
_output_shapes
:??*
dtype0
?
4Adam/simple_model2/larger_conv_block/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64Adam/simple_model2/larger_conv_block/conv2d_5/bias/v
?
HAdam/simple_model2/larger_conv_block/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOp4Adam/simple_model2/larger_conv_block/conv2d_5/bias/v*
_output_shapes	
:?*
dtype0
?
6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*G
shared_name86Adam/simple_model2/larger_conv_block/conv2d_6/kernel/v
?
JAdam/simple_model2/larger_conv_block/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/v*(
_output_shapes
:??*
dtype0
?
4Adam/simple_model2/larger_conv_block/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64Adam/simple_model2/larger_conv_block/conv2d_6/bias/v
?
HAdam/simple_model2/larger_conv_block/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOp4Adam/simple_model2/larger_conv_block/conv2d_6/bias/v*
_output_shapes	
:?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*D
shared_name53Adam/simple_model2/prediction_block2/dense/kernel/v
?
GAdam/simple_model2/prediction_block2/dense/kernel/v/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
1Adam/simple_model2/prediction_block2/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31Adam/simple_model2/prediction_block2/dense/bias/v
?
EAdam/simple_model2/prediction_block2/dense/bias/v/Read/ReadVariableOpReadVariableOp1Adam/simple_model2/prediction_block2/dense/bias/v*
_output_shapes	
:?*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*F
shared_name75Adam/simple_model2/prediction_block2/dense_1/kernel/v
?
IAdam/simple_model2/prediction_block2/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53Adam/simple_model2/prediction_block2/dense_1/bias/v
?
GAdam/simple_model2/prediction_block2/dense_1/bias/v/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_2/kernel/v
?
IAdam/simple_model2/prediction_block2/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_2/bias/v
?
GAdam/simple_model2/prediction_block2/dense_2/bias/v/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_2/bias/v*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_3/kernel/v
?
IAdam/simple_model2/prediction_block2/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_3/bias/v
?
GAdam/simple_model2/prediction_block2/dense_3/bias/v/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_3/bias/v*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_4/kernel/v
?
IAdam/simple_model2/prediction_block2/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_4/kernel/v*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_4/bias/v
?
GAdam/simple_model2/prediction_block2/dense_4/bias/v/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_4/bias/v*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_5/kernel/v
?
IAdam/simple_model2/prediction_block2/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_5/kernel/v*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_5/bias/v
?
GAdam/simple_model2/prediction_block2/dense_5/bias/v/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_5/bias/v*
_output_shapes
:*
dtype0
?
5Adam/simple_model2/prediction_block2/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75Adam/simple_model2/prediction_block2/dense_6/kernel/v
?
IAdam/simple_model2/prediction_block2/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/simple_model2/prediction_block2/dense_6/kernel/v*
_output_shapes
:	?*
dtype0
?
3Adam/simple_model2/prediction_block2/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/simple_model2/prediction_block2/dense_6/bias/v
?
GAdam/simple_model2/prediction_block2/dense_6/bias/v/Read/ReadVariableOpReadVariableOp3Adam/simple_model2/prediction_block2/dense_6/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_1
block_1
block_2
block_3

prediction
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
?
layer_1
layer_2
layer_3
layer_4
layer_5
layer_6
	variables
trainable_variables
regularization_losses
	keras_api
?
layer_1
layer_2
layer_3
layer_4
layer_5
layer_6
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?
$layer_1
%layer_2
&layer_3
'layer_4
(layer_5
)layer_6
*layer_7
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?
/layer_1
0layer_2
1layer_3
2layer_4
3layer_5
4layer_6
5layer_7
6layer_8
7layer_9
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?
<iter

=beta_1

>beta_2
	?decay
@learning_rateAm?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17
S18
T19
U20
V21
W22
X23
Y24
Z25
[26
\27
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17
S18
T19
U20
V21
W22
X23
Y24
Z25
[26
\27
 
?
]layer_regularization_losses
^non_trainable_variables
	variables
trainable_variables
	regularization_losses
_layer_metrics
`metrics

alayers
 
 
 
 
?
blayer_regularization_losses
cnon_trainable_variables
	variables
trainable_variables
regularization_losses
dlayer_metrics
emetrics

flayers
h

Akernel
Bbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
R
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
h

Ckernel
Dbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
R
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
R
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api

A0
B1
C2
D3

A0
B1
C2
D3
 
?
layer_regularization_losses
?non_trainable_variables
	variables
trainable_variables
regularization_losses
?layer_metrics
?metrics
?layers
l

Ekernel
Fbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Gkernel
Hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

E0
F1
G2
H3

E0
F1
G2
H3
 
?
 ?layer_regularization_losses
?non_trainable_variables
 	variables
!trainable_variables
"regularization_losses
?layer_metrics
?metrics
?layers
l

Ikernel
Jbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Kkernel
Lbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Mkernel
Nbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
I0
J1
K2
L3
M4
N5
*
I0
J1
K2
L3
M4
N5
 
?
 ?layer_regularization_losses
?non_trainable_variables
+	variables
,trainable_variables
-regularization_losses
?layer_metrics
?metrics
?layers
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
}
?
activation

Okernel
Pbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
}
?
activation

Qkernel
Rbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Skernel
Tbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Ukernel
Vbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Wkernel
Xbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Ykernel
Zbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

[kernel
\bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
f
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
f
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
 
?
 ?layer_regularization_losses
?non_trainable_variables
8	variables
9trainable_variables
:regularization_losses
?layer_metrics
?metrics
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&simple_model2/conv_block/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$simple_model2/conv_block/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(simple_model2/conv_block/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&simple_model2/conv_block/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*simple_model2/conv_block_1/conv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(simple_model2/conv_block_1/conv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*simple_model2/conv_block_1/conv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(simple_model2/conv_block_1/conv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_model2/larger_conv_block/conv2d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-simple_model2/larger_conv_block/conv2d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/simple_model2/larger_conv_block/conv2d_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-simple_model2/larger_conv_block/conv2d_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/simple_model2/larger_conv_block/conv2d_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-simple_model2/larger_conv_block/conv2d_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,simple_model2/prediction_block2/dense/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*simple_model2/prediction_block2/dense/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.simple_model2/prediction_block2/dense_1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,simple_model2/prediction_block2/dense_1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.simple_model2/prediction_block2/dense_2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,simple_model2/prediction_block2/dense_2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.simple_model2/prediction_block2/dense_3/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,simple_model2/prediction_block2/dense_3/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.simple_model2/prediction_block2/dense_4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,simple_model2/prediction_block2/dense_4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.simple_model2/prediction_block2/dense_5/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,simple_model2/prediction_block2/dense_5/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.simple_model2/prediction_block2/dense_6/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,simple_model2/prediction_block2/dense_6/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
Y
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
#
0
1
2
3
4
 
 
 
 
 

A0
B1

A0
B1
 
?
 ?layer_regularization_losses
?non_trainable_variables
g	variables
htrainable_variables
iregularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
k	variables
ltrainable_variables
mregularization_losses
?layer_metrics
?metrics
?layers

C0
D1

C0
D1
 
?
 ?layer_regularization_losses
?non_trainable_variables
o	variables
ptrainable_variables
qregularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
s	variables
ttrainable_variables
uregularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
w	variables
xtrainable_variables
yregularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
{	variables
|trainable_variables
}regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
 
*
0
1
2
3
4
5

E0
F1

E0
F1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

G0
H1

G0
H1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
 
*
0
1
2
3
4
5

I0
J1

I0
J1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

K0
L1

K0
L1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

M0
N1

M0
N1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
 
1
$0
%1
&2
'3
(4
)5
*6
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

O0
P1

O0
P1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

Q0
R1

Q0
R1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

S0
T1

S0
T1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

U0
V1

U0
V1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

W0
X1

W0
X1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

Y0
Z1

Y0
Z1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers

[0
\1

[0
\1
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
 
?
/0
01
12
23
34
45
56
67
78
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
 

?0
 
 
 
 
 
 
 
 
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
 
 
 
 

?0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUE-Adam/simple_model2/conv_block/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/simple_model2/conv_block/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/simple_model2/conv_block/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/simple_model2/conv_block/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/simple_model2/conv_block_1/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/simple_model2/conv_block_1/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/simple_model2/conv_block_1/conv2d_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/simple_model2/conv_block_1/conv2d_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_model2/larger_conv_block/conv2d_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_model2/larger_conv_block/conv2d_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_model2/larger_conv_block/conv2d_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/simple_model2/prediction_block2/dense/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_1/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_1/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_2/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_2/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_3/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_3/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_4/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_4/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_5/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_5/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_6/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_6/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/simple_model2/conv_block/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/simple_model2/conv_block/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/simple_model2/conv_block/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/simple_model2/conv_block/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/simple_model2/conv_block_1/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/simple_model2/conv_block_1/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/simple_model2/conv_block_1/conv2d_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/simple_model2/conv_block_1/conv2d_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_model2/larger_conv_block/conv2d_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_model2/larger_conv_block/conv2d_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/simple_model2/larger_conv_block/conv2d_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/simple_model2/prediction_block2/dense/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_1/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_1/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_2/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_2/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_3/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_3/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_4/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_4/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_5/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_5/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/simple_model2/prediction_block2/dense_6/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/simple_model2/prediction_block2/dense_6/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1&simple_model2/conv_block/conv2d/kernel$simple_model2/conv_block/conv2d/bias(simple_model2/conv_block/conv2d_1/kernel&simple_model2/conv_block/conv2d_1/bias*simple_model2/conv_block_1/conv2d_2/kernel(simple_model2/conv_block_1/conv2d_2/bias*simple_model2/conv_block_1/conv2d_3/kernel(simple_model2/conv_block_1/conv2d_3/bias/simple_model2/larger_conv_block/conv2d_4/kernel-simple_model2/larger_conv_block/conv2d_4/bias/simple_model2/larger_conv_block/conv2d_5/kernel-simple_model2/larger_conv_block/conv2d_5/bias/simple_model2/larger_conv_block/conv2d_6/kernel-simple_model2/larger_conv_block/conv2d_6/bias,simple_model2/prediction_block2/dense/kernel*simple_model2/prediction_block2/dense/bias.simple_model2/prediction_block2/dense_1/kernel,simple_model2/prediction_block2/dense_1/bias.simple_model2/prediction_block2/dense_2/kernel,simple_model2/prediction_block2/dense_2/bias.simple_model2/prediction_block2/dense_3/kernel,simple_model2/prediction_block2/dense_3/bias.simple_model2/prediction_block2/dense_4/kernel,simple_model2/prediction_block2/dense_4/bias.simple_model2/prediction_block2/dense_5/kernel,simple_model2/prediction_block2/dense_5/bias.simple_model2/prediction_block2/dense_6/kernel,simple_model2/prediction_block2/dense_6/bias*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_126836
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?7
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp:simple_model2/conv_block/conv2d/kernel/Read/ReadVariableOp8simple_model2/conv_block/conv2d/bias/Read/ReadVariableOp<simple_model2/conv_block/conv2d_1/kernel/Read/ReadVariableOp:simple_model2/conv_block/conv2d_1/bias/Read/ReadVariableOp>simple_model2/conv_block_1/conv2d_2/kernel/Read/ReadVariableOp<simple_model2/conv_block_1/conv2d_2/bias/Read/ReadVariableOp>simple_model2/conv_block_1/conv2d_3/kernel/Read/ReadVariableOp<simple_model2/conv_block_1/conv2d_3/bias/Read/ReadVariableOpCsimple_model2/larger_conv_block/conv2d_4/kernel/Read/ReadVariableOpAsimple_model2/larger_conv_block/conv2d_4/bias/Read/ReadVariableOpCsimple_model2/larger_conv_block/conv2d_5/kernel/Read/ReadVariableOpAsimple_model2/larger_conv_block/conv2d_5/bias/Read/ReadVariableOpCsimple_model2/larger_conv_block/conv2d_6/kernel/Read/ReadVariableOpAsimple_model2/larger_conv_block/conv2d_6/bias/Read/ReadVariableOp@simple_model2/prediction_block2/dense/kernel/Read/ReadVariableOp>simple_model2/prediction_block2/dense/bias/Read/ReadVariableOpBsimple_model2/prediction_block2/dense_1/kernel/Read/ReadVariableOp@simple_model2/prediction_block2/dense_1/bias/Read/ReadVariableOpBsimple_model2/prediction_block2/dense_2/kernel/Read/ReadVariableOp@simple_model2/prediction_block2/dense_2/bias/Read/ReadVariableOpBsimple_model2/prediction_block2/dense_3/kernel/Read/ReadVariableOp@simple_model2/prediction_block2/dense_3/bias/Read/ReadVariableOpBsimple_model2/prediction_block2/dense_4/kernel/Read/ReadVariableOp@simple_model2/prediction_block2/dense_4/bias/Read/ReadVariableOpBsimple_model2/prediction_block2/dense_5/kernel/Read/ReadVariableOp@simple_model2/prediction_block2/dense_5/bias/Read/ReadVariableOpBsimple_model2/prediction_block2/dense_6/kernel/Read/ReadVariableOp@simple_model2/prediction_block2/dense_6/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOpAAdam/simple_model2/conv_block/conv2d/kernel/m/Read/ReadVariableOp?Adam/simple_model2/conv_block/conv2d/bias/m/Read/ReadVariableOpCAdam/simple_model2/conv_block/conv2d_1/kernel/m/Read/ReadVariableOpAAdam/simple_model2/conv_block/conv2d_1/bias/m/Read/ReadVariableOpEAdam/simple_model2/conv_block_1/conv2d_2/kernel/m/Read/ReadVariableOpCAdam/simple_model2/conv_block_1/conv2d_2/bias/m/Read/ReadVariableOpEAdam/simple_model2/conv_block_1/conv2d_3/kernel/m/Read/ReadVariableOpCAdam/simple_model2/conv_block_1/conv2d_3/bias/m/Read/ReadVariableOpJAdam/simple_model2/larger_conv_block/conv2d_4/kernel/m/Read/ReadVariableOpHAdam/simple_model2/larger_conv_block/conv2d_4/bias/m/Read/ReadVariableOpJAdam/simple_model2/larger_conv_block/conv2d_5/kernel/m/Read/ReadVariableOpHAdam/simple_model2/larger_conv_block/conv2d_5/bias/m/Read/ReadVariableOpJAdam/simple_model2/larger_conv_block/conv2d_6/kernel/m/Read/ReadVariableOpHAdam/simple_model2/larger_conv_block/conv2d_6/bias/m/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense/kernel/m/Read/ReadVariableOpEAdam/simple_model2/prediction_block2/dense/bias/m/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_1/kernel/m/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_1/bias/m/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_2/kernel/m/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_2/bias/m/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_3/kernel/m/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_3/bias/m/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_4/kernel/m/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_4/bias/m/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_5/kernel/m/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_5/bias/m/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_6/kernel/m/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_6/bias/m/Read/ReadVariableOpAAdam/simple_model2/conv_block/conv2d/kernel/v/Read/ReadVariableOp?Adam/simple_model2/conv_block/conv2d/bias/v/Read/ReadVariableOpCAdam/simple_model2/conv_block/conv2d_1/kernel/v/Read/ReadVariableOpAAdam/simple_model2/conv_block/conv2d_1/bias/v/Read/ReadVariableOpEAdam/simple_model2/conv_block_1/conv2d_2/kernel/v/Read/ReadVariableOpCAdam/simple_model2/conv_block_1/conv2d_2/bias/v/Read/ReadVariableOpEAdam/simple_model2/conv_block_1/conv2d_3/kernel/v/Read/ReadVariableOpCAdam/simple_model2/conv_block_1/conv2d_3/bias/v/Read/ReadVariableOpJAdam/simple_model2/larger_conv_block/conv2d_4/kernel/v/Read/ReadVariableOpHAdam/simple_model2/larger_conv_block/conv2d_4/bias/v/Read/ReadVariableOpJAdam/simple_model2/larger_conv_block/conv2d_5/kernel/v/Read/ReadVariableOpHAdam/simple_model2/larger_conv_block/conv2d_5/bias/v/Read/ReadVariableOpJAdam/simple_model2/larger_conv_block/conv2d_6/kernel/v/Read/ReadVariableOpHAdam/simple_model2/larger_conv_block/conv2d_6/bias/v/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense/kernel/v/Read/ReadVariableOpEAdam/simple_model2/prediction_block2/dense/bias/v/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_1/kernel/v/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_1/bias/v/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_2/kernel/v/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_2/bias/v/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_3/kernel/v/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_3/bias/v/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_4/kernel/v/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_4/bias/v/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_5/kernel/v/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_5/bias/v/Read/ReadVariableOpIAdam/simple_model2/prediction_block2/dense_6/kernel/v/Read/ReadVariableOpGAdam/simple_model2/prediction_block2/dense_6/bias/v/Read/ReadVariableOpConst*|
Tinu
s2q	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_128017
?&
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate&simple_model2/conv_block/conv2d/kernel$simple_model2/conv_block/conv2d/bias(simple_model2/conv_block/conv2d_1/kernel&simple_model2/conv_block/conv2d_1/bias*simple_model2/conv_block_1/conv2d_2/kernel(simple_model2/conv_block_1/conv2d_2/bias*simple_model2/conv_block_1/conv2d_3/kernel(simple_model2/conv_block_1/conv2d_3/bias/simple_model2/larger_conv_block/conv2d_4/kernel-simple_model2/larger_conv_block/conv2d_4/bias/simple_model2/larger_conv_block/conv2d_5/kernel-simple_model2/larger_conv_block/conv2d_5/bias/simple_model2/larger_conv_block/conv2d_6/kernel-simple_model2/larger_conv_block/conv2d_6/bias,simple_model2/prediction_block2/dense/kernel*simple_model2/prediction_block2/dense/bias.simple_model2/prediction_block2/dense_1/kernel,simple_model2/prediction_block2/dense_1/bias.simple_model2/prediction_block2/dense_2/kernel,simple_model2/prediction_block2/dense_2/bias.simple_model2/prediction_block2/dense_3/kernel,simple_model2/prediction_block2/dense_3/bias.simple_model2/prediction_block2/dense_4/kernel,simple_model2/prediction_block2/dense_4/bias.simple_model2/prediction_block2/dense_5/kernel,simple_model2/prediction_block2/dense_5/bias.simple_model2/prediction_block2/dense_6/kernel,simple_model2/prediction_block2/dense_6/biastotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6total_7count_7total_8count_8total_9count_9total_10count_10-Adam/simple_model2/conv_block/conv2d/kernel/m+Adam/simple_model2/conv_block/conv2d/bias/m/Adam/simple_model2/conv_block/conv2d_1/kernel/m-Adam/simple_model2/conv_block/conv2d_1/bias/m1Adam/simple_model2/conv_block_1/conv2d_2/kernel/m/Adam/simple_model2/conv_block_1/conv2d_2/bias/m1Adam/simple_model2/conv_block_1/conv2d_3/kernel/m/Adam/simple_model2/conv_block_1/conv2d_3/bias/m6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/m4Adam/simple_model2/larger_conv_block/conv2d_4/bias/m6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/m4Adam/simple_model2/larger_conv_block/conv2d_5/bias/m6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/m4Adam/simple_model2/larger_conv_block/conv2d_6/bias/m3Adam/simple_model2/prediction_block2/dense/kernel/m1Adam/simple_model2/prediction_block2/dense/bias/m5Adam/simple_model2/prediction_block2/dense_1/kernel/m3Adam/simple_model2/prediction_block2/dense_1/bias/m5Adam/simple_model2/prediction_block2/dense_2/kernel/m3Adam/simple_model2/prediction_block2/dense_2/bias/m5Adam/simple_model2/prediction_block2/dense_3/kernel/m3Adam/simple_model2/prediction_block2/dense_3/bias/m5Adam/simple_model2/prediction_block2/dense_4/kernel/m3Adam/simple_model2/prediction_block2/dense_4/bias/m5Adam/simple_model2/prediction_block2/dense_5/kernel/m3Adam/simple_model2/prediction_block2/dense_5/bias/m5Adam/simple_model2/prediction_block2/dense_6/kernel/m3Adam/simple_model2/prediction_block2/dense_6/bias/m-Adam/simple_model2/conv_block/conv2d/kernel/v+Adam/simple_model2/conv_block/conv2d/bias/v/Adam/simple_model2/conv_block/conv2d_1/kernel/v-Adam/simple_model2/conv_block/conv2d_1/bias/v1Adam/simple_model2/conv_block_1/conv2d_2/kernel/v/Adam/simple_model2/conv_block_1/conv2d_2/bias/v1Adam/simple_model2/conv_block_1/conv2d_3/kernel/v/Adam/simple_model2/conv_block_1/conv2d_3/bias/v6Adam/simple_model2/larger_conv_block/conv2d_4/kernel/v4Adam/simple_model2/larger_conv_block/conv2d_4/bias/v6Adam/simple_model2/larger_conv_block/conv2d_5/kernel/v4Adam/simple_model2/larger_conv_block/conv2d_5/bias/v6Adam/simple_model2/larger_conv_block/conv2d_6/kernel/v4Adam/simple_model2/larger_conv_block/conv2d_6/bias/v3Adam/simple_model2/prediction_block2/dense/kernel/v1Adam/simple_model2/prediction_block2/dense/bias/v5Adam/simple_model2/prediction_block2/dense_1/kernel/v3Adam/simple_model2/prediction_block2/dense_1/bias/v5Adam/simple_model2/prediction_block2/dense_2/kernel/v3Adam/simple_model2/prediction_block2/dense_2/bias/v5Adam/simple_model2/prediction_block2/dense_3/kernel/v3Adam/simple_model2/prediction_block2/dense_3/bias/v5Adam/simple_model2/prediction_block2/dense_4/kernel/v3Adam/simple_model2/prediction_block2/dense_4/bias/v5Adam/simple_model2/prediction_block2/dense_5/kernel/v3Adam/simple_model2/prediction_block2/dense_5/bias/v5Adam/simple_model2/prediction_block2/dense_6/kernel/v3Adam/simple_model2/prediction_block2/dense_6/bias/v*{
Tint
r2p*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_128360??
?
?
2__inference_prediction_block2_layer_call_fn_127489

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:
	unknown_9:	?

unknown_10:

unknown_11:	?

unknown_12:
identity

identity_1

identity_2

identity_3

identity_4??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_prediction_block2_layer_call_and_return_conditional_losses_1259282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_126330

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?b
?
M__inference_prediction_block2_layer_call_and_return_conditional_losses_127657

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:9
&dense_6_matmul_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddw
dense/re_lu/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/re_lu/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const?
dropout_3/dropout/MulMuldense/re_lu/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeShapedense/re_lu/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAdd?
dense_1/re_lu_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/re_lu_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Softmax?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Softmax?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Softmax?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddy
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_6/Softmax?
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_3/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitydense_4/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_5/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitydense_6/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?>
__inference__traced_save_128017
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopE
Asavev2_simple_model2_conv_block_conv2d_kernel_read_readvariableopC
?savev2_simple_model2_conv_block_conv2d_bias_read_readvariableopG
Csavev2_simple_model2_conv_block_conv2d_1_kernel_read_readvariableopE
Asavev2_simple_model2_conv_block_conv2d_1_bias_read_readvariableopI
Esavev2_simple_model2_conv_block_1_conv2d_2_kernel_read_readvariableopG
Csavev2_simple_model2_conv_block_1_conv2d_2_bias_read_readvariableopI
Esavev2_simple_model2_conv_block_1_conv2d_3_kernel_read_readvariableopG
Csavev2_simple_model2_conv_block_1_conv2d_3_bias_read_readvariableopN
Jsavev2_simple_model2_larger_conv_block_conv2d_4_kernel_read_readvariableopL
Hsavev2_simple_model2_larger_conv_block_conv2d_4_bias_read_readvariableopN
Jsavev2_simple_model2_larger_conv_block_conv2d_5_kernel_read_readvariableopL
Hsavev2_simple_model2_larger_conv_block_conv2d_5_bias_read_readvariableopN
Jsavev2_simple_model2_larger_conv_block_conv2d_6_kernel_read_readvariableopL
Hsavev2_simple_model2_larger_conv_block_conv2d_6_bias_read_readvariableopK
Gsavev2_simple_model2_prediction_block2_dense_kernel_read_readvariableopI
Esavev2_simple_model2_prediction_block2_dense_bias_read_readvariableopM
Isavev2_simple_model2_prediction_block2_dense_1_kernel_read_readvariableopK
Gsavev2_simple_model2_prediction_block2_dense_1_bias_read_readvariableopM
Isavev2_simple_model2_prediction_block2_dense_2_kernel_read_readvariableopK
Gsavev2_simple_model2_prediction_block2_dense_2_bias_read_readvariableopM
Isavev2_simple_model2_prediction_block2_dense_3_kernel_read_readvariableopK
Gsavev2_simple_model2_prediction_block2_dense_3_bias_read_readvariableopM
Isavev2_simple_model2_prediction_block2_dense_4_kernel_read_readvariableopK
Gsavev2_simple_model2_prediction_block2_dense_4_bias_read_readvariableopM
Isavev2_simple_model2_prediction_block2_dense_5_kernel_read_readvariableopK
Gsavev2_simple_model2_prediction_block2_dense_5_bias_read_readvariableopM
Isavev2_simple_model2_prediction_block2_dense_6_kernel_read_readvariableopK
Gsavev2_simple_model2_prediction_block2_dense_6_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_7_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_8_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_9_read_readvariableop&
"savev2_count_9_read_readvariableop'
#savev2_total_10_read_readvariableop'
#savev2_count_10_read_readvariableopL
Hsavev2_adam_simple_model2_conv_block_conv2d_kernel_m_read_readvariableopJ
Fsavev2_adam_simple_model2_conv_block_conv2d_bias_m_read_readvariableopN
Jsavev2_adam_simple_model2_conv_block_conv2d_1_kernel_m_read_readvariableopL
Hsavev2_adam_simple_model2_conv_block_conv2d_1_bias_m_read_readvariableopP
Lsavev2_adam_simple_model2_conv_block_1_conv2d_2_kernel_m_read_readvariableopN
Jsavev2_adam_simple_model2_conv_block_1_conv2d_2_bias_m_read_readvariableopP
Lsavev2_adam_simple_model2_conv_block_1_conv2d_3_kernel_m_read_readvariableopN
Jsavev2_adam_simple_model2_conv_block_1_conv2d_3_bias_m_read_readvariableopU
Qsavev2_adam_simple_model2_larger_conv_block_conv2d_4_kernel_m_read_readvariableopS
Osavev2_adam_simple_model2_larger_conv_block_conv2d_4_bias_m_read_readvariableopU
Qsavev2_adam_simple_model2_larger_conv_block_conv2d_5_kernel_m_read_readvariableopS
Osavev2_adam_simple_model2_larger_conv_block_conv2d_5_bias_m_read_readvariableopU
Qsavev2_adam_simple_model2_larger_conv_block_conv2d_6_kernel_m_read_readvariableopS
Osavev2_adam_simple_model2_larger_conv_block_conv2d_6_bias_m_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_kernel_m_read_readvariableopP
Lsavev2_adam_simple_model2_prediction_block2_dense_bias_m_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_1_kernel_m_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_1_bias_m_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_2_kernel_m_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_2_bias_m_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_3_kernel_m_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_3_bias_m_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_4_kernel_m_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_4_bias_m_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_5_kernel_m_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_5_bias_m_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_6_kernel_m_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_6_bias_m_read_readvariableopL
Hsavev2_adam_simple_model2_conv_block_conv2d_kernel_v_read_readvariableopJ
Fsavev2_adam_simple_model2_conv_block_conv2d_bias_v_read_readvariableopN
Jsavev2_adam_simple_model2_conv_block_conv2d_1_kernel_v_read_readvariableopL
Hsavev2_adam_simple_model2_conv_block_conv2d_1_bias_v_read_readvariableopP
Lsavev2_adam_simple_model2_conv_block_1_conv2d_2_kernel_v_read_readvariableopN
Jsavev2_adam_simple_model2_conv_block_1_conv2d_2_bias_v_read_readvariableopP
Lsavev2_adam_simple_model2_conv_block_1_conv2d_3_kernel_v_read_readvariableopN
Jsavev2_adam_simple_model2_conv_block_1_conv2d_3_bias_v_read_readvariableopU
Qsavev2_adam_simple_model2_larger_conv_block_conv2d_4_kernel_v_read_readvariableopS
Osavev2_adam_simple_model2_larger_conv_block_conv2d_4_bias_v_read_readvariableopU
Qsavev2_adam_simple_model2_larger_conv_block_conv2d_5_kernel_v_read_readvariableopS
Osavev2_adam_simple_model2_larger_conv_block_conv2d_5_bias_v_read_readvariableopU
Qsavev2_adam_simple_model2_larger_conv_block_conv2d_6_kernel_v_read_readvariableopS
Osavev2_adam_simple_model2_larger_conv_block_conv2d_6_bias_v_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_kernel_v_read_readvariableopP
Lsavev2_adam_simple_model2_prediction_block2_dense_bias_v_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_1_kernel_v_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_1_bias_v_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_2_kernel_v_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_2_bias_v_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_3_kernel_v_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_3_bias_v_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_4_kernel_v_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_4_bias_v_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_5_kernel_v_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_5_bias_v_read_readvariableopT
Psavev2_adam_simple_model2_prediction_block2_dense_6_kernel_v_read_readvariableopR
Nsavev2_adam_simple_model2_prediction_block2_dense_6_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?3
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?2
value?2B?2pB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?
value?B?pB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?<
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopAsavev2_simple_model2_conv_block_conv2d_kernel_read_readvariableop?savev2_simple_model2_conv_block_conv2d_bias_read_readvariableopCsavev2_simple_model2_conv_block_conv2d_1_kernel_read_readvariableopAsavev2_simple_model2_conv_block_conv2d_1_bias_read_readvariableopEsavev2_simple_model2_conv_block_1_conv2d_2_kernel_read_readvariableopCsavev2_simple_model2_conv_block_1_conv2d_2_bias_read_readvariableopEsavev2_simple_model2_conv_block_1_conv2d_3_kernel_read_readvariableopCsavev2_simple_model2_conv_block_1_conv2d_3_bias_read_readvariableopJsavev2_simple_model2_larger_conv_block_conv2d_4_kernel_read_readvariableopHsavev2_simple_model2_larger_conv_block_conv2d_4_bias_read_readvariableopJsavev2_simple_model2_larger_conv_block_conv2d_5_kernel_read_readvariableopHsavev2_simple_model2_larger_conv_block_conv2d_5_bias_read_readvariableopJsavev2_simple_model2_larger_conv_block_conv2d_6_kernel_read_readvariableopHsavev2_simple_model2_larger_conv_block_conv2d_6_bias_read_readvariableopGsavev2_simple_model2_prediction_block2_dense_kernel_read_readvariableopEsavev2_simple_model2_prediction_block2_dense_bias_read_readvariableopIsavev2_simple_model2_prediction_block2_dense_1_kernel_read_readvariableopGsavev2_simple_model2_prediction_block2_dense_1_bias_read_readvariableopIsavev2_simple_model2_prediction_block2_dense_2_kernel_read_readvariableopGsavev2_simple_model2_prediction_block2_dense_2_bias_read_readvariableopIsavev2_simple_model2_prediction_block2_dense_3_kernel_read_readvariableopGsavev2_simple_model2_prediction_block2_dense_3_bias_read_readvariableopIsavev2_simple_model2_prediction_block2_dense_4_kernel_read_readvariableopGsavev2_simple_model2_prediction_block2_dense_4_bias_read_readvariableopIsavev2_simple_model2_prediction_block2_dense_5_kernel_read_readvariableopGsavev2_simple_model2_prediction_block2_dense_5_bias_read_readvariableopIsavev2_simple_model2_prediction_block2_dense_6_kernel_read_readvariableopGsavev2_simple_model2_prediction_block2_dense_6_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_7_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_8_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_9_read_readvariableop"savev2_count_9_read_readvariableop#savev2_total_10_read_readvariableop#savev2_count_10_read_readvariableopHsavev2_adam_simple_model2_conv_block_conv2d_kernel_m_read_readvariableopFsavev2_adam_simple_model2_conv_block_conv2d_bias_m_read_readvariableopJsavev2_adam_simple_model2_conv_block_conv2d_1_kernel_m_read_readvariableopHsavev2_adam_simple_model2_conv_block_conv2d_1_bias_m_read_readvariableopLsavev2_adam_simple_model2_conv_block_1_conv2d_2_kernel_m_read_readvariableopJsavev2_adam_simple_model2_conv_block_1_conv2d_2_bias_m_read_readvariableopLsavev2_adam_simple_model2_conv_block_1_conv2d_3_kernel_m_read_readvariableopJsavev2_adam_simple_model2_conv_block_1_conv2d_3_bias_m_read_readvariableopQsavev2_adam_simple_model2_larger_conv_block_conv2d_4_kernel_m_read_readvariableopOsavev2_adam_simple_model2_larger_conv_block_conv2d_4_bias_m_read_readvariableopQsavev2_adam_simple_model2_larger_conv_block_conv2d_5_kernel_m_read_readvariableopOsavev2_adam_simple_model2_larger_conv_block_conv2d_5_bias_m_read_readvariableopQsavev2_adam_simple_model2_larger_conv_block_conv2d_6_kernel_m_read_readvariableopOsavev2_adam_simple_model2_larger_conv_block_conv2d_6_bias_m_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_kernel_m_read_readvariableopLsavev2_adam_simple_model2_prediction_block2_dense_bias_m_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_1_kernel_m_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_1_bias_m_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_2_kernel_m_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_2_bias_m_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_3_kernel_m_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_3_bias_m_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_4_kernel_m_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_4_bias_m_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_5_kernel_m_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_5_bias_m_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_6_kernel_m_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_6_bias_m_read_readvariableopHsavev2_adam_simple_model2_conv_block_conv2d_kernel_v_read_readvariableopFsavev2_adam_simple_model2_conv_block_conv2d_bias_v_read_readvariableopJsavev2_adam_simple_model2_conv_block_conv2d_1_kernel_v_read_readvariableopHsavev2_adam_simple_model2_conv_block_conv2d_1_bias_v_read_readvariableopLsavev2_adam_simple_model2_conv_block_1_conv2d_2_kernel_v_read_readvariableopJsavev2_adam_simple_model2_conv_block_1_conv2d_2_bias_v_read_readvariableopLsavev2_adam_simple_model2_conv_block_1_conv2d_3_kernel_v_read_readvariableopJsavev2_adam_simple_model2_conv_block_1_conv2d_3_bias_v_read_readvariableopQsavev2_adam_simple_model2_larger_conv_block_conv2d_4_kernel_v_read_readvariableopOsavev2_adam_simple_model2_larger_conv_block_conv2d_4_bias_v_read_readvariableopQsavev2_adam_simple_model2_larger_conv_block_conv2d_5_kernel_v_read_readvariableopOsavev2_adam_simple_model2_larger_conv_block_conv2d_5_bias_v_read_readvariableopQsavev2_adam_simple_model2_larger_conv_block_conv2d_6_kernel_v_read_readvariableopOsavev2_adam_simple_model2_larger_conv_block_conv2d_6_bias_v_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_kernel_v_read_readvariableopLsavev2_adam_simple_model2_prediction_block2_dense_bias_v_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_1_kernel_v_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_1_bias_v_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_2_kernel_v_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_2_bias_v_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_3_kernel_v_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_3_bias_v_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_4_kernel_v_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_4_bias_v_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_5_kernel_v_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_5_bias_v_read_readvariableopPsavev2_adam_simple_model2_prediction_block2_dense_6_kernel_v_read_readvariableopNsavev2_adam_simple_model2_prediction_block2_dense_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *~
dtypest
r2p	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : :  : : @:@:@@:@:@?:?:??:?:??:?:
??:?:
??:?:	?::	?::	?::	?::	?:: : : : : : : : : : : : : : : : : : : : : : : : :  : : @:@:@@:@:@?:?:??:?:??:?:
??:?:
??:?:	?::	?::	?::	?::	?:: : :  : : @:@:@@:@:@?:?:??:?:??:?:
??:?:
??:?:	?::	?::	?::	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 	

_output_shapes
: :,
(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::% !

_output_shapes
:	?: !

_output_shapes
::"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :,8(
&
_output_shapes
: : 9

_output_shapes
: :,:(
&
_output_shapes
:  : ;

_output_shapes
: :,<(
&
_output_shapes
: @: =

_output_shapes
:@:,>(
&
_output_shapes
:@@: ?

_output_shapes
:@:-@)
'
_output_shapes
:@?:!A

_output_shapes	
:?:.B*
(
_output_shapes
:??:!C

_output_shapes	
:?:.D*
(
_output_shapes
:??:!E

_output_shapes	
:?:&F"
 
_output_shapes
:
??:!G

_output_shapes	
:?:&H"
 
_output_shapes
:
??:!I

_output_shapes	
:?:%J!

_output_shapes
:	?: K

_output_shapes
::%L!

_output_shapes
:	?: M

_output_shapes
::%N!

_output_shapes
:	?: O

_output_shapes
::%P!

_output_shapes
:	?: Q

_output_shapes
::%R!

_output_shapes
:	?: S

_output_shapes
::,T(
&
_output_shapes
: : U

_output_shapes
: :,V(
&
_output_shapes
:  : W

_output_shapes
: :,X(
&
_output_shapes
: @: Y

_output_shapes
:@:,Z(
&
_output_shapes
:@@: [

_output_shapes
:@:-\)
'
_output_shapes
:@?:!]

_output_shapes	
:?:.^*
(
_output_shapes
:??:!_

_output_shapes	
:?:.`*
(
_output_shapes
:??:!a

_output_shapes	
:?:&b"
 
_output_shapes
:
??:!c

_output_shapes	
:?:&d"
 
_output_shapes
:
??:!e

_output_shapes	
:?:%f!

_output_shapes
:	?: g

_output_shapes
::%h!

_output_shapes
:	?: i

_output_shapes
::%j!

_output_shapes
:	?: k

_output_shapes
::%l!

_output_shapes
:	?: m

_output_shapes
::%n!

_output_shapes
:	?: o

_output_shapes
::p

_output_shapes
: 
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_127259

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?#
?
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_125854

inputsB
'conv2d_4_conv2d_readvariableop_resource:@?7
(conv2d_4_biasadd_readvariableop_resource:	?C
'conv2d_5_conv2d_readvariableop_resource:??7
(conv2d_5_biasadd_readvariableop_resource:	?C
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d_4/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*0
_output_shapes
:?????????		?*
alpha%???>2
leaky_re_lu_4/LeakyRelu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D%leaky_re_lu_4/LeakyRelu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_5/LeakyRelu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D%leaky_re_lu_5/LeakyRelu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_6/LeakyRelu?
max_pooling2d_2/MaxPoolMaxPool%leaky_re_lu_6/LeakyRelu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
IdentityIdentity max_pooling2d_2/MaxPool:output:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@: : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?#
?
H__inference_conv_block_1_layer_call_and_return_conditional_losses_127405

inputsA
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_2/LeakyRelu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_3/LeakyRelu?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul max_pooling2d_1/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_2/dropout/Mul_1?
IdentityIdentitydropout_2/dropout/Mul_1:z:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?#
?
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_127448

inputsB
'conv2d_4_conv2d_readvariableop_resource:@?7
(conv2d_4_biasadd_readvariableop_resource:	?C
'conv2d_5_conv2d_readvariableop_resource:??7
(conv2d_5_biasadd_readvariableop_resource:	?C
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?
identity??conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d_4/BiasAdd?
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*0
_output_shapes
:?????????		?*
alpha%???>2
leaky_re_lu_4/LeakyRelu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D%leaky_re_lu_4/LeakyRelu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_5/LeakyRelu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D%leaky_re_lu_5/LeakyRelu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_6/LeakyRelu?
max_pooling2d_2/MaxPoolMaxPool%leaky_re_lu_6/LeakyRelu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
IdentityIdentity max_pooling2d_2/MaxPool:output:0 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@: : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_simple_model2_layer_call_fn_126034
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:	?

unknown_22:

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:
identity

identity_1

identity_2

identity_3

identity_4??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model2_layer_call_and_return_conditional_losses_1259672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
.__inference_simple_model2_layer_call_fn_126905

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:	?

unknown_22:

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:
identity

identity_1

identity_2

identity_3

identity_4??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model2_layer_call_and_return_conditional_losses_1259672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
I__inference_simple_model2_layer_call_and_return_conditional_losses_127089

inputsJ
0conv_block_conv2d_conv2d_readvariableop_resource: ?
1conv_block_conv2d_biasadd_readvariableop_resource: L
2conv_block_conv2d_1_conv2d_readvariableop_resource:  A
3conv_block_conv2d_1_biasadd_readvariableop_resource: N
4conv_block_1_conv2d_2_conv2d_readvariableop_resource: @C
5conv_block_1_conv2d_2_biasadd_readvariableop_resource:@N
4conv_block_1_conv2d_3_conv2d_readvariableop_resource:@@C
5conv_block_1_conv2d_3_biasadd_readvariableop_resource:@T
9larger_conv_block_conv2d_4_conv2d_readvariableop_resource:@?I
:larger_conv_block_conv2d_4_biasadd_readvariableop_resource:	?U
9larger_conv_block_conv2d_5_conv2d_readvariableop_resource:??I
:larger_conv_block_conv2d_5_biasadd_readvariableop_resource:	?U
9larger_conv_block_conv2d_6_conv2d_readvariableop_resource:??I
:larger_conv_block_conv2d_6_biasadd_readvariableop_resource:	?J
6prediction_block2_dense_matmul_readvariableop_resource:
??F
7prediction_block2_dense_biasadd_readvariableop_resource:	?L
8prediction_block2_dense_1_matmul_readvariableop_resource:
??H
9prediction_block2_dense_1_biasadd_readvariableop_resource:	?K
8prediction_block2_dense_2_matmul_readvariableop_resource:	?G
9prediction_block2_dense_2_biasadd_readvariableop_resource:K
8prediction_block2_dense_3_matmul_readvariableop_resource:	?G
9prediction_block2_dense_3_biasadd_readvariableop_resource:K
8prediction_block2_dense_4_matmul_readvariableop_resource:	?G
9prediction_block2_dense_4_biasadd_readvariableop_resource:K
8prediction_block2_dense_5_matmul_readvariableop_resource:	?G
9prediction_block2_dense_5_biasadd_readvariableop_resource:K
8prediction_block2_dense_6_matmul_readvariableop_resource:	?G
9prediction_block2_dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4??(conv_block/conv2d/BiasAdd/ReadVariableOp?'conv_block/conv2d/Conv2D/ReadVariableOp?*conv_block/conv2d_1/BiasAdd/ReadVariableOp?)conv_block/conv2d_1/Conv2D/ReadVariableOp?,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp?+conv_block_1/conv2d_2/Conv2D/ReadVariableOp?,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp?+conv_block_1/conv2d_3/Conv2D/ReadVariableOp?1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp?1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp?1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp?.prediction_block2/dense/BiasAdd/ReadVariableOp?-prediction_block2/dense/MatMul/ReadVariableOp?0prediction_block2/dense_1/BiasAdd/ReadVariableOp?/prediction_block2/dense_1/MatMul/ReadVariableOp?0prediction_block2/dense_2/BiasAdd/ReadVariableOp?/prediction_block2/dense_2/MatMul/ReadVariableOp?0prediction_block2/dense_3/BiasAdd/ReadVariableOp?/prediction_block2/dense_3/MatMul/ReadVariableOp?0prediction_block2/dense_4/BiasAdd/ReadVariableOp?/prediction_block2/dense_4/MatMul/ReadVariableOp?0prediction_block2/dense_5/BiasAdd/ReadVariableOp?/prediction_block2/dense_5/MatMul/ReadVariableOp?0prediction_block2/dense_6/BiasAdd/ReadVariableOp?/prediction_block2/dense_6/MatMul/ReadVariableOpr
dropout/IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@@2
dropout/Identity?
'conv_block/conv2d/Conv2D/ReadVariableOpReadVariableOp0conv_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'conv_block/conv2d/Conv2D/ReadVariableOp?
conv_block/conv2d/Conv2DConv2Ddropout/Identity:output:0/conv_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingVALID*
strides
2
conv_block/conv2d/Conv2D?
(conv_block/conv2d/BiasAdd/ReadVariableOpReadVariableOp1conv_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(conv_block/conv2d/BiasAdd/ReadVariableOp?
conv_block/conv2d/BiasAddBiasAdd!conv_block/conv2d/Conv2D:output:00conv_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv_block/conv2d/BiasAdd?
 conv_block/leaky_re_lu/LeakyRelu	LeakyRelu"conv_block/conv2d/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2"
 conv_block/leaky_re_lu/LeakyRelu?
)conv_block/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2conv_block_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02+
)conv_block/conv2d_1/Conv2D/ReadVariableOp?
conv_block/conv2d_1/Conv2DConv2D.conv_block/leaky_re_lu/LeakyRelu:activations:01conv_block/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingSAME*
strides
2
conv_block/conv2d_1/Conv2D?
*conv_block/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3conv_block_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv_block/conv2d_1/BiasAdd/ReadVariableOp?
conv_block/conv2d_1/BiasAddBiasAdd#conv_block/conv2d_1/Conv2D:output:02conv_block/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv_block/conv2d_1/BiasAdd?
"conv_block/leaky_re_lu_1/LeakyRelu	LeakyRelu$conv_block/conv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2$
"conv_block/leaky_re_lu_1/LeakyRelu?
 conv_block/max_pooling2d/MaxPoolMaxPool0conv_block/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2"
 conv_block/max_pooling2d/MaxPool?
conv_block/dropout_1/IdentityIdentity)conv_block/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
conv_block/dropout_1/Identity?
+conv_block_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4conv_block_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+conv_block_1/conv2d_2/Conv2D/ReadVariableOp?
conv_block_1/conv2d_2/Conv2DConv2D&conv_block/dropout_1/Identity:output:03conv_block_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv_block_1/conv2d_2/Conv2D?
,conv_block_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5conv_block_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp?
conv_block_1/conv2d_2/BiasAddBiasAdd%conv_block_1/conv2d_2/Conv2D:output:04conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv_block_1/conv2d_2/BiasAdd?
$conv_block_1/leaky_re_lu_2/LeakyRelu	LeakyRelu&conv_block_1/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2&
$conv_block_1/leaky_re_lu_2/LeakyRelu?
+conv_block_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4conv_block_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+conv_block_1/conv2d_3/Conv2D/ReadVariableOp?
conv_block_1/conv2d_3/Conv2DConv2D2conv_block_1/leaky_re_lu_2/LeakyRelu:activations:03conv_block_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv_block_1/conv2d_3/Conv2D?
,conv_block_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5conv_block_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp?
conv_block_1/conv2d_3/BiasAddBiasAdd%conv_block_1/conv2d_3/Conv2D:output:04conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv_block_1/conv2d_3/BiasAdd?
$conv_block_1/leaky_re_lu_3/LeakyRelu	LeakyRelu&conv_block_1/conv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2&
$conv_block_1/leaky_re_lu_3/LeakyRelu?
$conv_block_1/max_pooling2d_1/MaxPoolMaxPool2conv_block_1/leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$conv_block_1/max_pooling2d_1/MaxPool?
conv_block_1/dropout_2/IdentityIdentity-conv_block_1/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2!
conv_block_1/dropout_2/Identity?
0larger_conv_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp9larger_conv_block_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype022
0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp?
!larger_conv_block/conv2d_4/Conv2DConv2D(conv_block_1/dropout_2/Identity:output:08larger_conv_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2#
!larger_conv_block/conv2d_4/Conv2D?
1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp:larger_conv_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?
"larger_conv_block/conv2d_4/BiasAddBiasAdd*larger_conv_block/conv2d_4/Conv2D:output:09larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2$
"larger_conv_block/conv2d_4/BiasAdd?
)larger_conv_block/leaky_re_lu_4/LeakyRelu	LeakyRelu+larger_conv_block/conv2d_4/BiasAdd:output:0*0
_output_shapes
:?????????		?*
alpha%???>2+
)larger_conv_block/leaky_re_lu_4/LeakyRelu?
0larger_conv_block/conv2d_5/Conv2D/ReadVariableOpReadVariableOp9larger_conv_block_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp?
!larger_conv_block/conv2d_5/Conv2DConv2D7larger_conv_block/leaky_re_lu_4/LeakyRelu:activations:08larger_conv_block/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2#
!larger_conv_block/conv2d_5/Conv2D?
1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp:larger_conv_block_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?
"larger_conv_block/conv2d_5/BiasAddBiasAdd*larger_conv_block/conv2d_5/Conv2D:output:09larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"larger_conv_block/conv2d_5/BiasAdd?
)larger_conv_block/leaky_re_lu_5/LeakyRelu	LeakyRelu+larger_conv_block/conv2d_5/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2+
)larger_conv_block/leaky_re_lu_5/LeakyRelu?
0larger_conv_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp9larger_conv_block_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp?
!larger_conv_block/conv2d_6/Conv2DConv2D7larger_conv_block/leaky_re_lu_5/LeakyRelu:activations:08larger_conv_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!larger_conv_block/conv2d_6/Conv2D?
1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp:larger_conv_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?
"larger_conv_block/conv2d_6/BiasAddBiasAdd*larger_conv_block/conv2d_6/Conv2D:output:09larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"larger_conv_block/conv2d_6/BiasAdd?
)larger_conv_block/leaky_re_lu_6/LeakyRelu	LeakyRelu+larger_conv_block/conv2d_6/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2+
)larger_conv_block/leaky_re_lu_6/LeakyRelu?
)larger_conv_block/max_pooling2d_2/MaxPoolMaxPool7larger_conv_block/leaky_re_lu_6/LeakyRelu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2+
)larger_conv_block/max_pooling2d_2/MaxPool?
prediction_block2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2!
prediction_block2/flatten/Const?
!prediction_block2/flatten/ReshapeReshape2larger_conv_block/max_pooling2d_2/MaxPool:output:0(prediction_block2/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2#
!prediction_block2/flatten/Reshape?
-prediction_block2/dense/MatMul/ReadVariableOpReadVariableOp6prediction_block2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-prediction_block2/dense/MatMul/ReadVariableOp?
prediction_block2/dense/MatMulMatMul*prediction_block2/flatten/Reshape:output:05prediction_block2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
prediction_block2/dense/MatMul?
.prediction_block2/dense/BiasAdd/ReadVariableOpReadVariableOp7prediction_block2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.prediction_block2/dense/BiasAdd/ReadVariableOp?
prediction_block2/dense/BiasAddBiasAdd(prediction_block2/dense/MatMul:product:06prediction_block2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
prediction_block2/dense/BiasAdd?
"prediction_block2/dense/re_lu/ReluRelu(prediction_block2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2$
"prediction_block2/dense/re_lu/Relu?
$prediction_block2/dropout_3/IdentityIdentity0prediction_block2/dense/re_lu/Relu:activations:0*
T0*(
_output_shapes
:??????????2&
$prediction_block2/dropout_3/Identity?
/prediction_block2/dense_1/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/prediction_block2/dense_1/MatMul/ReadVariableOp?
 prediction_block2/dense_1/MatMulMatMul-prediction_block2/dropout_3/Identity:output:07prediction_block2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 prediction_block2/dense_1/MatMul?
0prediction_block2/dense_1/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0prediction_block2/dense_1/BiasAdd/ReadVariableOp?
!prediction_block2/dense_1/BiasAddBiasAdd*prediction_block2/dense_1/MatMul:product:08prediction_block2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!prediction_block2/dense_1/BiasAdd?
&prediction_block2/dense_1/re_lu_1/ReluRelu*prediction_block2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&prediction_block2/dense_1/re_lu_1/Relu?
/prediction_block2/dense_2/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_2/MatMul/ReadVariableOp?
 prediction_block2/dense_2/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_2/MatMul?
0prediction_block2/dense_2/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_2/BiasAdd/ReadVariableOp?
!prediction_block2/dense_2/BiasAddBiasAdd*prediction_block2/dense_2/MatMul:product:08prediction_block2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_2/BiasAdd?
!prediction_block2/dense_2/SoftmaxSoftmax*prediction_block2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_2/Softmax?
/prediction_block2/dense_3/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_3/MatMul/ReadVariableOp?
 prediction_block2/dense_3/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_3/MatMul?
0prediction_block2/dense_3/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_3/BiasAdd/ReadVariableOp?
!prediction_block2/dense_3/BiasAddBiasAdd*prediction_block2/dense_3/MatMul:product:08prediction_block2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_3/BiasAdd?
!prediction_block2/dense_3/SoftmaxSoftmax*prediction_block2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_3/Softmax?
/prediction_block2/dense_4/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_4/MatMul/ReadVariableOp?
 prediction_block2/dense_4/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_4/MatMul?
0prediction_block2/dense_4/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_4/BiasAdd/ReadVariableOp?
!prediction_block2/dense_4/BiasAddBiasAdd*prediction_block2/dense_4/MatMul:product:08prediction_block2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_4/BiasAdd?
!prediction_block2/dense_4/SoftmaxSoftmax*prediction_block2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_4/Softmax?
/prediction_block2/dense_5/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_5/MatMul/ReadVariableOp?
 prediction_block2/dense_5/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_5/MatMul?
0prediction_block2/dense_5/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_5/BiasAdd/ReadVariableOp?
!prediction_block2/dense_5/BiasAddBiasAdd*prediction_block2/dense_5/MatMul:product:08prediction_block2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_5/BiasAdd?
!prediction_block2/dense_5/SoftmaxSoftmax*prediction_block2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_5/Softmax?
/prediction_block2/dense_6/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_6/MatMul/ReadVariableOp?
 prediction_block2/dense_6/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_6/MatMul?
0prediction_block2/dense_6/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_6/BiasAdd/ReadVariableOp?
!prediction_block2/dense_6/BiasAddBiasAdd*prediction_block2/dense_6/MatMul:product:08prediction_block2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_6/BiasAdd?
!prediction_block2/dense_6/SoftmaxSoftmax*prediction_block2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_6/Softmax?
IdentityIdentity+prediction_block2/dense_2/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity+prediction_block2/dense_3/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+prediction_block2/dense_4/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity+prediction_block2/dense_5/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity+prediction_block2/dense_6/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(conv_block/conv2d/BiasAdd/ReadVariableOp(conv_block/conv2d/BiasAdd/ReadVariableOp2R
'conv_block/conv2d/Conv2D/ReadVariableOp'conv_block/conv2d/Conv2D/ReadVariableOp2X
*conv_block/conv2d_1/BiasAdd/ReadVariableOp*conv_block/conv2d_1/BiasAdd/ReadVariableOp2V
)conv_block/conv2d_1/Conv2D/ReadVariableOp)conv_block/conv2d_1/Conv2D/ReadVariableOp2\
,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+conv_block_1/conv2d_2/Conv2D/ReadVariableOp+conv_block_1/conv2d_2/Conv2D/ReadVariableOp2\
,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+conv_block_1/conv2d_3/Conv2D/ReadVariableOp+conv_block_1/conv2d_3/Conv2D/ReadVariableOp2f
1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp2d
0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2f
1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp2d
0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2f
1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp2d
0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp2`
.prediction_block2/dense/BiasAdd/ReadVariableOp.prediction_block2/dense/BiasAdd/ReadVariableOp2^
-prediction_block2/dense/MatMul/ReadVariableOp-prediction_block2/dense/MatMul/ReadVariableOp2d
0prediction_block2/dense_1/BiasAdd/ReadVariableOp0prediction_block2/dense_1/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_1/MatMul/ReadVariableOp/prediction_block2/dense_1/MatMul/ReadVariableOp2d
0prediction_block2/dense_2/BiasAdd/ReadVariableOp0prediction_block2/dense_2/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_2/MatMul/ReadVariableOp/prediction_block2/dense_2/MatMul/ReadVariableOp2d
0prediction_block2/dense_3/BiasAdd/ReadVariableOp0prediction_block2/dense_3/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_3/MatMul/ReadVariableOp/prediction_block2/dense_3/MatMul/ReadVariableOp2d
0prediction_block2/dense_4/BiasAdd/ReadVariableOp0prediction_block2/dense_4/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_4/MatMul/ReadVariableOp/prediction_block2/dense_4/MatMul/ReadVariableOp2d
0prediction_block2/dense_5/BiasAdd/ReadVariableOp0prediction_block2/dense_5/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_5/MatMul/ReadVariableOp/prediction_block2/dense_5/MatMul/ReadVariableOp2d
0prediction_block2/dense_6/BiasAdd/ReadVariableOp0prediction_block2/dense_6/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_6/MatMul/ReadVariableOp/prediction_block2/dense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_125766

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
-__inference_conv_block_1_layer_call_fn_127345

inputs!
unknown: @
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_1_layer_call_and_return_conditional_losses_1258182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?6
?
I__inference_simple_model2_layer_call_and_return_conditional_losses_126759
input_1+
conv_block_126690: 
conv_block_126692: +
conv_block_126694:  
conv_block_126696: -
conv_block_1_126699: @!
conv_block_1_126701:@-
conv_block_1_126703:@@!
conv_block_1_126705:@3
larger_conv_block_126708:@?'
larger_conv_block_126710:	?4
larger_conv_block_126712:??'
larger_conv_block_126714:	?4
larger_conv_block_126716:??'
larger_conv_block_126718:	?,
prediction_block2_126721:
??'
prediction_block2_126723:	?,
prediction_block2_126725:
??'
prediction_block2_126727:	?+
prediction_block2_126729:	?&
prediction_block2_126731:+
prediction_block2_126733:	?&
prediction_block2_126735:+
prediction_block2_126737:	?&
prediction_block2_126739:+
prediction_block2_126741:	?&
prediction_block2_126743:+
prediction_block2_126745:	?&
prediction_block2_126747:
identity

identity_1

identity_2

identity_3

identity_4??"conv_block/StatefulPartitionedCall?$conv_block_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?)larger_conv_block/StatefulPartitionedCall?)prediction_block2/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1263302!
dropout/StatefulPartitionedCall?
"conv_block/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv_block_126690conv_block_126692conv_block_126694conv_block_126696*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_block_layer_call_and_return_conditional_losses_1262992$
"conv_block/StatefulPartitionedCall?
$conv_block_1/StatefulPartitionedCallStatefulPartitionedCall+conv_block/StatefulPartitionedCall:output:0conv_block_1_126699conv_block_1_126701conv_block_1_126703conv_block_1_126705*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_1_layer_call_and_return_conditional_losses_1262452&
$conv_block_1/StatefulPartitionedCall?
)larger_conv_block/StatefulPartitionedCallStatefulPartitionedCall-conv_block_1/StatefulPartitionedCall:output:0larger_conv_block_126708larger_conv_block_126710larger_conv_block_126712larger_conv_block_126714larger_conv_block_126716larger_conv_block_126718*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_1258542+
)larger_conv_block/StatefulPartitionedCall?
)prediction_block2/StatefulPartitionedCallStatefulPartitionedCall2larger_conv_block/StatefulPartitionedCall:output:0prediction_block2_126721prediction_block2_126723prediction_block2_126725prediction_block2_126727prediction_block2_126729prediction_block2_126731prediction_block2_126733prediction_block2_126735prediction_block2_126737prediction_block2_126739prediction_block2_126741prediction_block2_126743prediction_block2_126745prediction_block2_126747*
Tin
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_prediction_block2_layer_call_and_return_conditional_losses_1261452+
)prediction_block2/StatefulPartitionedCall?
IdentityIdentity2prediction_block2/StatefulPartitionedCall:output:0#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity2prediction_block2/StatefulPartitionedCall:output:1#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity2prediction_block2/StatefulPartitionedCall:output:2#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity2prediction_block2/StatefulPartitionedCall:output:3#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity2prediction_block2/StatefulPartitionedCall:output:4#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"conv_block/StatefulPartitionedCall"conv_block/StatefulPartitionedCall2L
$conv_block_1/StatefulPartitionedCall$conv_block_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2V
)larger_conv_block/StatefulPartitionedCall)larger_conv_block/StatefulPartitionedCall2V
)prediction_block2/StatefulPartitionedCall)prediction_block2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?#
?
H__inference_conv_block_1_layer_call_and_return_conditional_losses_126245

inputsA
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_2/LeakyRelu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_3/LeakyRelu?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul max_pooling2d_1/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_2/dropout/Mul_1?
IdentityIdentitydropout_2/dropout/Mul_1:z:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
2__inference_larger_conv_block_layer_call_fn_127422

inputs"
unknown:@?
	unknown_0:	?%
	unknown_1:??
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_1258542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_126836
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:	?

unknown_22:

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:
identity

identity_1

identity_2

identity_3

identity_4??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1257182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_125724

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?Y
?
M__inference_prediction_block2_layer_call_and_return_conditional_losses_127590

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:9
&dense_6_matmul_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddw
dense/re_lu/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/re_lu/Relu?
dropout_3/IdentityIdentitydense/re_lu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_3/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_3/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAdd?
dense_1/re_lu_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/re_lu_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Softmax?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Softmax?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Softmax?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddy
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_6/Softmax?
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_3/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitydense_4/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_5/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitydense_6/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?#
?
F__inference_conv_block_layer_call_and_return_conditional_losses_127332

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: 
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu/LeakyRelu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D#leaky_re_lu/LeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu_1/LeakyRelu?
max_pooling2d/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulmax_pooling2d/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_1/dropout/Mul_1?
IdentityIdentitydropout_1/dropout/Mul_1:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_125736

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?#
?
F__inference_conv_block_layer_call_and_return_conditional_losses_126299

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: 
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu/LeakyRelu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D#leaky_re_lu/LeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu_1/LeakyRelu?
max_pooling2d/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulmax_pooling2d/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_1/dropout/Mul_1?
IdentityIdentitydropout_1/dropout/Mul_1:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?"
!__inference__wrapped_model_125718
input_1X
>simple_model2_conv_block_conv2d_conv2d_readvariableop_resource: M
?simple_model2_conv_block_conv2d_biasadd_readvariableop_resource: Z
@simple_model2_conv_block_conv2d_1_conv2d_readvariableop_resource:  O
Asimple_model2_conv_block_conv2d_1_biasadd_readvariableop_resource: \
Bsimple_model2_conv_block_1_conv2d_2_conv2d_readvariableop_resource: @Q
Csimple_model2_conv_block_1_conv2d_2_biasadd_readvariableop_resource:@\
Bsimple_model2_conv_block_1_conv2d_3_conv2d_readvariableop_resource:@@Q
Csimple_model2_conv_block_1_conv2d_3_biasadd_readvariableop_resource:@b
Gsimple_model2_larger_conv_block_conv2d_4_conv2d_readvariableop_resource:@?W
Hsimple_model2_larger_conv_block_conv2d_4_biasadd_readvariableop_resource:	?c
Gsimple_model2_larger_conv_block_conv2d_5_conv2d_readvariableop_resource:??W
Hsimple_model2_larger_conv_block_conv2d_5_biasadd_readvariableop_resource:	?c
Gsimple_model2_larger_conv_block_conv2d_6_conv2d_readvariableop_resource:??W
Hsimple_model2_larger_conv_block_conv2d_6_biasadd_readvariableop_resource:	?X
Dsimple_model2_prediction_block2_dense_matmul_readvariableop_resource:
??T
Esimple_model2_prediction_block2_dense_biasadd_readvariableop_resource:	?Z
Fsimple_model2_prediction_block2_dense_1_matmul_readvariableop_resource:
??V
Gsimple_model2_prediction_block2_dense_1_biasadd_readvariableop_resource:	?Y
Fsimple_model2_prediction_block2_dense_2_matmul_readvariableop_resource:	?U
Gsimple_model2_prediction_block2_dense_2_biasadd_readvariableop_resource:Y
Fsimple_model2_prediction_block2_dense_3_matmul_readvariableop_resource:	?U
Gsimple_model2_prediction_block2_dense_3_biasadd_readvariableop_resource:Y
Fsimple_model2_prediction_block2_dense_4_matmul_readvariableop_resource:	?U
Gsimple_model2_prediction_block2_dense_4_biasadd_readvariableop_resource:Y
Fsimple_model2_prediction_block2_dense_5_matmul_readvariableop_resource:	?U
Gsimple_model2_prediction_block2_dense_5_biasadd_readvariableop_resource:Y
Fsimple_model2_prediction_block2_dense_6_matmul_readvariableop_resource:	?U
Gsimple_model2_prediction_block2_dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4??6simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp?5simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp?8simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp?7simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp?:simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp?9simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp?:simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp?9simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp??simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?>simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp??simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?>simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp??simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?>simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp?<simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp?;simple_model2/prediction_block2/dense/MatMul/ReadVariableOp?>simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp?=simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp?>simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp?=simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp?>simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp?=simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp?>simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp?=simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp?>simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp?=simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp?>simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp?=simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp?
simple_model2/dropout/IdentityIdentityinput_1*
T0*/
_output_shapes
:?????????@@2 
simple_model2/dropout/Identity?
5simple_model2/conv_block/conv2d/Conv2D/ReadVariableOpReadVariableOp>simple_model2_conv_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp?
&simple_model2/conv_block/conv2d/Conv2DConv2D'simple_model2/dropout/Identity:output:0=simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingVALID*
strides
2(
&simple_model2/conv_block/conv2d/Conv2D?
6simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOpReadVariableOp?simple_model2_conv_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp?
'simple_model2/conv_block/conv2d/BiasAddBiasAdd/simple_model2/conv_block/conv2d/Conv2D:output:0>simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2)
'simple_model2/conv_block/conv2d/BiasAdd?
.simple_model2/conv_block/leaky_re_lu/LeakyRelu	LeakyRelu0simple_model2/conv_block/conv2d/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>20
.simple_model2/conv_block/leaky_re_lu/LeakyRelu?
7simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@simple_model2_conv_block_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype029
7simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp?
(simple_model2/conv_block/conv2d_1/Conv2DConv2D<simple_model2/conv_block/leaky_re_lu/LeakyRelu:activations:0?simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingSAME*
strides
2*
(simple_model2/conv_block/conv2d_1/Conv2D?
8simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAsimple_model2_conv_block_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp?
)simple_model2/conv_block/conv2d_1/BiasAddBiasAdd1simple_model2/conv_block/conv2d_1/Conv2D:output:0@simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2+
)simple_model2/conv_block/conv2d_1/BiasAdd?
0simple_model2/conv_block/leaky_re_lu_1/LeakyRelu	LeakyRelu2simple_model2/conv_block/conv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>22
0simple_model2/conv_block/leaky_re_lu_1/LeakyRelu?
.simple_model2/conv_block/max_pooling2d/MaxPoolMaxPool>simple_model2/conv_block/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
20
.simple_model2/conv_block/max_pooling2d/MaxPool?
+simple_model2/conv_block/dropout_1/IdentityIdentity7simple_model2/conv_block/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2-
+simple_model2/conv_block/dropout_1/Identity?
9simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOpBsimple_model2_conv_block_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02;
9simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp?
*simple_model2/conv_block_1/conv2d_2/Conv2DConv2D4simple_model2/conv_block/dropout_1/Identity:output:0Asimple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2,
*simple_model2/conv_block_1/conv2d_2/Conv2D?
:simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpCsimple_model2_conv_block_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp?
+simple_model2/conv_block_1/conv2d_2/BiasAddBiasAdd3simple_model2/conv_block_1/conv2d_2/Conv2D:output:0Bsimple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2-
+simple_model2/conv_block_1/conv2d_2/BiasAdd?
2simple_model2/conv_block_1/leaky_re_lu_2/LeakyRelu	LeakyRelu4simple_model2/conv_block_1/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>24
2simple_model2/conv_block_1/leaky_re_lu_2/LeakyRelu?
9simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOpBsimple_model2_conv_block_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp?
*simple_model2/conv_block_1/conv2d_3/Conv2DConv2D@simple_model2/conv_block_1/leaky_re_lu_2/LeakyRelu:activations:0Asimple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2,
*simple_model2/conv_block_1/conv2d_3/Conv2D?
:simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpCsimple_model2_conv_block_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp?
+simple_model2/conv_block_1/conv2d_3/BiasAddBiasAdd3simple_model2/conv_block_1/conv2d_3/Conv2D:output:0Bsimple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2-
+simple_model2/conv_block_1/conv2d_3/BiasAdd?
2simple_model2/conv_block_1/leaky_re_lu_3/LeakyRelu	LeakyRelu4simple_model2/conv_block_1/conv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>24
2simple_model2/conv_block_1/leaky_re_lu_3/LeakyRelu?
2simple_model2/conv_block_1/max_pooling2d_1/MaxPoolMaxPool@simple_model2/conv_block_1/leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
24
2simple_model2/conv_block_1/max_pooling2d_1/MaxPool?
-simple_model2/conv_block_1/dropout_2/IdentityIdentity;simple_model2/conv_block_1/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2/
-simple_model2/conv_block_1/dropout_2/Identity?
>simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOpGsimple_model2_larger_conv_block_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02@
>simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp?
/simple_model2/larger_conv_block/conv2d_4/Conv2DConv2D6simple_model2/conv_block_1/dropout_2/Identity:output:0Fsimple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
21
/simple_model2/larger_conv_block/conv2d_4/Conv2D?
?simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpHsimple_model2_larger_conv_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?
0simple_model2/larger_conv_block/conv2d_4/BiasAddBiasAdd8simple_model2/larger_conv_block/conv2d_4/Conv2D:output:0Gsimple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?22
0simple_model2/larger_conv_block/conv2d_4/BiasAdd?
7simple_model2/larger_conv_block/leaky_re_lu_4/LeakyRelu	LeakyRelu9simple_model2/larger_conv_block/conv2d_4/BiasAdd:output:0*0
_output_shapes
:?????????		?*
alpha%???>29
7simple_model2/larger_conv_block/leaky_re_lu_4/LeakyRelu?
>simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOpReadVariableOpGsimple_model2_larger_conv_block_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02@
>simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp?
/simple_model2/larger_conv_block/conv2d_5/Conv2DConv2DEsimple_model2/larger_conv_block/leaky_re_lu_4/LeakyRelu:activations:0Fsimple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
21
/simple_model2/larger_conv_block/conv2d_5/Conv2D?
?simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpHsimple_model2_larger_conv_block_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?
0simple_model2/larger_conv_block/conv2d_5/BiasAddBiasAdd8simple_model2/larger_conv_block/conv2d_5/Conv2D:output:0Gsimple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????22
0simple_model2/larger_conv_block/conv2d_5/BiasAdd?
7simple_model2/larger_conv_block/leaky_re_lu_5/LeakyRelu	LeakyRelu9simple_model2/larger_conv_block/conv2d_5/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>29
7simple_model2/larger_conv_block/leaky_re_lu_5/LeakyRelu?
>simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOpGsimple_model2_larger_conv_block_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02@
>simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp?
/simple_model2/larger_conv_block/conv2d_6/Conv2DConv2DEsimple_model2/larger_conv_block/leaky_re_lu_5/LeakyRelu:activations:0Fsimple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
21
/simple_model2/larger_conv_block/conv2d_6/Conv2D?
?simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpHsimple_model2_larger_conv_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?
0simple_model2/larger_conv_block/conv2d_6/BiasAddBiasAdd8simple_model2/larger_conv_block/conv2d_6/Conv2D:output:0Gsimple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????22
0simple_model2/larger_conv_block/conv2d_6/BiasAdd?
7simple_model2/larger_conv_block/leaky_re_lu_6/LeakyRelu	LeakyRelu9simple_model2/larger_conv_block/conv2d_6/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>29
7simple_model2/larger_conv_block/leaky_re_lu_6/LeakyRelu?
7simple_model2/larger_conv_block/max_pooling2d_2/MaxPoolMaxPoolEsimple_model2/larger_conv_block/leaky_re_lu_6/LeakyRelu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
29
7simple_model2/larger_conv_block/max_pooling2d_2/MaxPool?
-simple_model2/prediction_block2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2/
-simple_model2/prediction_block2/flatten/Const?
/simple_model2/prediction_block2/flatten/ReshapeReshape@simple_model2/larger_conv_block/max_pooling2d_2/MaxPool:output:06simple_model2/prediction_block2/flatten/Const:output:0*
T0*(
_output_shapes
:??????????21
/simple_model2/prediction_block2/flatten/Reshape?
;simple_model2/prediction_block2/dense/MatMul/ReadVariableOpReadVariableOpDsimple_model2_prediction_block2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;simple_model2/prediction_block2/dense/MatMul/ReadVariableOp?
,simple_model2/prediction_block2/dense/MatMulMatMul8simple_model2/prediction_block2/flatten/Reshape:output:0Csimple_model2/prediction_block2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,simple_model2/prediction_block2/dense/MatMul?
<simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOpReadVariableOpEsimple_model2_prediction_block2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp?
-simple_model2/prediction_block2/dense/BiasAddBiasAdd6simple_model2/prediction_block2/dense/MatMul:product:0Dsimple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-simple_model2/prediction_block2/dense/BiasAdd?
0simple_model2/prediction_block2/dense/re_lu/ReluRelu6simple_model2/prediction_block2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????22
0simple_model2/prediction_block2/dense/re_lu/Relu?
2simple_model2/prediction_block2/dropout_3/IdentityIdentity>simple_model2/prediction_block2/dense/re_lu/Relu:activations:0*
T0*(
_output_shapes
:??????????24
2simple_model2/prediction_block2/dropout_3/Identity?
=simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOpReadVariableOpFsimple_model2_prediction_block2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02?
=simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp?
.simple_model2/prediction_block2/dense_1/MatMulMatMul;simple_model2/prediction_block2/dropout_3/Identity:output:0Esimple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.simple_model2/prediction_block2/dense_1/MatMul?
>simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOpReadVariableOpGsimple_model2_prediction_block2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp?
/simple_model2/prediction_block2/dense_1/BiasAddBiasAdd8simple_model2/prediction_block2/dense_1/MatMul:product:0Fsimple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/simple_model2/prediction_block2/dense_1/BiasAdd?
4simple_model2/prediction_block2/dense_1/re_lu_1/ReluRelu8simple_model2/prediction_block2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????26
4simple_model2/prediction_block2/dense_1/re_lu_1/Relu?
=simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOpReadVariableOpFsimple_model2_prediction_block2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02?
=simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp?
.simple_model2/prediction_block2/dense_2/MatMulMatMulBsimple_model2/prediction_block2/dense_1/re_lu_1/Relu:activations:0Esimple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.simple_model2/prediction_block2/dense_2/MatMul?
>simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOpReadVariableOpGsimple_model2_prediction_block2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp?
/simple_model2/prediction_block2/dense_2/BiasAddBiasAdd8simple_model2/prediction_block2/dense_2/MatMul:product:0Fsimple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_2/BiasAdd?
/simple_model2/prediction_block2/dense_2/SoftmaxSoftmax8simple_model2/prediction_block2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_2/Softmax?
=simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOpReadVariableOpFsimple_model2_prediction_block2_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02?
=simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp?
.simple_model2/prediction_block2/dense_3/MatMulMatMulBsimple_model2/prediction_block2/dense_1/re_lu_1/Relu:activations:0Esimple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.simple_model2/prediction_block2/dense_3/MatMul?
>simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOpReadVariableOpGsimple_model2_prediction_block2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp?
/simple_model2/prediction_block2/dense_3/BiasAddBiasAdd8simple_model2/prediction_block2/dense_3/MatMul:product:0Fsimple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_3/BiasAdd?
/simple_model2/prediction_block2/dense_3/SoftmaxSoftmax8simple_model2/prediction_block2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_3/Softmax?
=simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOpReadVariableOpFsimple_model2_prediction_block2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02?
=simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp?
.simple_model2/prediction_block2/dense_4/MatMulMatMulBsimple_model2/prediction_block2/dense_1/re_lu_1/Relu:activations:0Esimple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.simple_model2/prediction_block2/dense_4/MatMul?
>simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOpReadVariableOpGsimple_model2_prediction_block2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp?
/simple_model2/prediction_block2/dense_4/BiasAddBiasAdd8simple_model2/prediction_block2/dense_4/MatMul:product:0Fsimple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_4/BiasAdd?
/simple_model2/prediction_block2/dense_4/SoftmaxSoftmax8simple_model2/prediction_block2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_4/Softmax?
=simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOpReadVariableOpFsimple_model2_prediction_block2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02?
=simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp?
.simple_model2/prediction_block2/dense_5/MatMulMatMulBsimple_model2/prediction_block2/dense_1/re_lu_1/Relu:activations:0Esimple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.simple_model2/prediction_block2/dense_5/MatMul?
>simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOpReadVariableOpGsimple_model2_prediction_block2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp?
/simple_model2/prediction_block2/dense_5/BiasAddBiasAdd8simple_model2/prediction_block2/dense_5/MatMul:product:0Fsimple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_5/BiasAdd?
/simple_model2/prediction_block2/dense_5/SoftmaxSoftmax8simple_model2/prediction_block2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_5/Softmax?
=simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOpReadVariableOpFsimple_model2_prediction_block2_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02?
=simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp?
.simple_model2/prediction_block2/dense_6/MatMulMatMulBsimple_model2/prediction_block2/dense_1/re_lu_1/Relu:activations:0Esimple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.simple_model2/prediction_block2/dense_6/MatMul?
>simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOpReadVariableOpGsimple_model2_prediction_block2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp?
/simple_model2/prediction_block2/dense_6/BiasAddBiasAdd8simple_model2/prediction_block2/dense_6/MatMul:product:0Fsimple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_6/BiasAdd?
/simple_model2/prediction_block2/dense_6/SoftmaxSoftmax8simple_model2/prediction_block2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????21
/simple_model2/prediction_block2/dense_6/Softmax?
IdentityIdentity9simple_model2/prediction_block2/dense_2/Softmax:softmax:07^simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp6^simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp9^simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp8^simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp=^simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp<^simple_model2/prediction_block2/dense/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity9simple_model2/prediction_block2/dense_3/Softmax:softmax:07^simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp6^simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp9^simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp8^simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp=^simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp<^simple_model2/prediction_block2/dense/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity9simple_model2/prediction_block2/dense_4/Softmax:softmax:07^simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp6^simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp9^simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp8^simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp=^simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp<^simple_model2/prediction_block2/dense/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity9simple_model2/prediction_block2/dense_5/Softmax:softmax:07^simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp6^simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp9^simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp8^simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp=^simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp<^simple_model2/prediction_block2/dense/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity9simple_model2/prediction_block2/dense_6/Softmax:softmax:07^simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp6^simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp9^simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp8^simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp;^simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:^simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp@^simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?^simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp=^simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp<^simple_model2/prediction_block2/dense/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp?^simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp>^simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp6simple_model2/conv_block/conv2d/BiasAdd/ReadVariableOp2n
5simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp5simple_model2/conv_block/conv2d/Conv2D/ReadVariableOp2t
8simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp8simple_model2/conv_block/conv2d_1/BiasAdd/ReadVariableOp2r
7simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp7simple_model2/conv_block/conv2d_1/Conv2D/ReadVariableOp2x
:simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:simple_model2/conv_block_1/conv2d_2/BiasAdd/ReadVariableOp2v
9simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp9simple_model2/conv_block_1/conv2d_2/Conv2D/ReadVariableOp2x
:simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:simple_model2/conv_block_1/conv2d_3/BiasAdd/ReadVariableOp2v
9simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp9simple_model2/conv_block_1/conv2d_3/Conv2D/ReadVariableOp2?
?simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?simple_model2/larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp2?
>simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp>simple_model2/larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2?
?simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?simple_model2/larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp2?
>simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp>simple_model2/larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2?
?simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?simple_model2/larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp2?
>simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp>simple_model2/larger_conv_block/conv2d_6/Conv2D/ReadVariableOp2|
<simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp<simple_model2/prediction_block2/dense/BiasAdd/ReadVariableOp2z
;simple_model2/prediction_block2/dense/MatMul/ReadVariableOp;simple_model2/prediction_block2/dense/MatMul/ReadVariableOp2?
>simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp>simple_model2/prediction_block2/dense_1/BiasAdd/ReadVariableOp2~
=simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp=simple_model2/prediction_block2/dense_1/MatMul/ReadVariableOp2?
>simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp>simple_model2/prediction_block2/dense_2/BiasAdd/ReadVariableOp2~
=simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp=simple_model2/prediction_block2/dense_2/MatMul/ReadVariableOp2?
>simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp>simple_model2/prediction_block2/dense_3/BiasAdd/ReadVariableOp2~
=simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp=simple_model2/prediction_block2/dense_3/MatMul/ReadVariableOp2?
>simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp>simple_model2/prediction_block2/dense_4/BiasAdd/ReadVariableOp2~
=simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp=simple_model2/prediction_block2/dense_4/MatMul/ReadVariableOp2?
>simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp>simple_model2/prediction_block2/dense_5/BiasAdd/ReadVariableOp2~
=simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp=simple_model2/prediction_block2/dense_5/MatMul/ReadVariableOp2?
>simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp>simple_model2/prediction_block2/dense_6/BiasAdd/ReadVariableOp2~
=simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp=simple_model2/prediction_block2/dense_6/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
-__inference_conv_block_1_layer_call_fn_127358

inputs!
unknown: @
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_1_layer_call_and_return_conditional_losses_1262452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?Y
?
M__inference_prediction_block2_layer_call_and_return_conditional_losses_125928

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:9
&dense_6_matmul_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddw
dense/re_lu/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/re_lu/Relu?
dropout_3/IdentityIdentitydense/re_lu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_3/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_3/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAdd?
dense_1/re_lu_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/re_lu_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Softmax?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Softmax?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Softmax?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddy
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_6/Softmax?
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_3/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitydense_4/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_5/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitydense_6/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv_block_layer_call_fn_127272

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_block_layer_call_and_return_conditional_losses_1257882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_127237

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1257662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
.__inference_simple_model2_layer_call_fn_126613
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:	?

unknown_22:

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:
identity

identity_1

identity_2

identity_3

identity_4??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model2_layer_call_and_return_conditional_losses_1264772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
??
?V
"__inference__traced_restore_128360
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: S
9assignvariableop_5_simple_model2_conv_block_conv2d_kernel: E
7assignvariableop_6_simple_model2_conv_block_conv2d_bias: U
;assignvariableop_7_simple_model2_conv_block_conv2d_1_kernel:  G
9assignvariableop_8_simple_model2_conv_block_conv2d_1_bias: W
=assignvariableop_9_simple_model2_conv_block_1_conv2d_2_kernel: @J
<assignvariableop_10_simple_model2_conv_block_1_conv2d_2_bias:@X
>assignvariableop_11_simple_model2_conv_block_1_conv2d_3_kernel:@@J
<assignvariableop_12_simple_model2_conv_block_1_conv2d_3_bias:@^
Cassignvariableop_13_simple_model2_larger_conv_block_conv2d_4_kernel:@?P
Aassignvariableop_14_simple_model2_larger_conv_block_conv2d_4_bias:	?_
Cassignvariableop_15_simple_model2_larger_conv_block_conv2d_5_kernel:??P
Aassignvariableop_16_simple_model2_larger_conv_block_conv2d_5_bias:	?_
Cassignvariableop_17_simple_model2_larger_conv_block_conv2d_6_kernel:??P
Aassignvariableop_18_simple_model2_larger_conv_block_conv2d_6_bias:	?T
@assignvariableop_19_simple_model2_prediction_block2_dense_kernel:
??M
>assignvariableop_20_simple_model2_prediction_block2_dense_bias:	?V
Bassignvariableop_21_simple_model2_prediction_block2_dense_1_kernel:
??O
@assignvariableop_22_simple_model2_prediction_block2_dense_1_bias:	?U
Bassignvariableop_23_simple_model2_prediction_block2_dense_2_kernel:	?N
@assignvariableop_24_simple_model2_prediction_block2_dense_2_bias:U
Bassignvariableop_25_simple_model2_prediction_block2_dense_3_kernel:	?N
@assignvariableop_26_simple_model2_prediction_block2_dense_3_bias:U
Bassignvariableop_27_simple_model2_prediction_block2_dense_4_kernel:	?N
@assignvariableop_28_simple_model2_prediction_block2_dense_4_bias:U
Bassignvariableop_29_simple_model2_prediction_block2_dense_5_kernel:	?N
@assignvariableop_30_simple_model2_prediction_block2_dense_5_bias:U
Bassignvariableop_31_simple_model2_prediction_block2_dense_6_kernel:	?N
@assignvariableop_32_simple_model2_prediction_block2_dense_6_bias:#
assignvariableop_33_total: #
assignvariableop_34_count: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: %
assignvariableop_37_total_2: %
assignvariableop_38_count_2: %
assignvariableop_39_total_3: %
assignvariableop_40_count_3: %
assignvariableop_41_total_4: %
assignvariableop_42_count_4: %
assignvariableop_43_total_5: %
assignvariableop_44_count_5: %
assignvariableop_45_total_6: %
assignvariableop_46_count_6: %
assignvariableop_47_total_7: %
assignvariableop_48_count_7: %
assignvariableop_49_total_8: %
assignvariableop_50_count_8: %
assignvariableop_51_total_9: %
assignvariableop_52_count_9: &
assignvariableop_53_total_10: &
assignvariableop_54_count_10: [
Aassignvariableop_55_adam_simple_model2_conv_block_conv2d_kernel_m: M
?assignvariableop_56_adam_simple_model2_conv_block_conv2d_bias_m: ]
Cassignvariableop_57_adam_simple_model2_conv_block_conv2d_1_kernel_m:  O
Aassignvariableop_58_adam_simple_model2_conv_block_conv2d_1_bias_m: _
Eassignvariableop_59_adam_simple_model2_conv_block_1_conv2d_2_kernel_m: @Q
Cassignvariableop_60_adam_simple_model2_conv_block_1_conv2d_2_bias_m:@_
Eassignvariableop_61_adam_simple_model2_conv_block_1_conv2d_3_kernel_m:@@Q
Cassignvariableop_62_adam_simple_model2_conv_block_1_conv2d_3_bias_m:@e
Jassignvariableop_63_adam_simple_model2_larger_conv_block_conv2d_4_kernel_m:@?W
Hassignvariableop_64_adam_simple_model2_larger_conv_block_conv2d_4_bias_m:	?f
Jassignvariableop_65_adam_simple_model2_larger_conv_block_conv2d_5_kernel_m:??W
Hassignvariableop_66_adam_simple_model2_larger_conv_block_conv2d_5_bias_m:	?f
Jassignvariableop_67_adam_simple_model2_larger_conv_block_conv2d_6_kernel_m:??W
Hassignvariableop_68_adam_simple_model2_larger_conv_block_conv2d_6_bias_m:	?[
Gassignvariableop_69_adam_simple_model2_prediction_block2_dense_kernel_m:
??T
Eassignvariableop_70_adam_simple_model2_prediction_block2_dense_bias_m:	?]
Iassignvariableop_71_adam_simple_model2_prediction_block2_dense_1_kernel_m:
??V
Gassignvariableop_72_adam_simple_model2_prediction_block2_dense_1_bias_m:	?\
Iassignvariableop_73_adam_simple_model2_prediction_block2_dense_2_kernel_m:	?U
Gassignvariableop_74_adam_simple_model2_prediction_block2_dense_2_bias_m:\
Iassignvariableop_75_adam_simple_model2_prediction_block2_dense_3_kernel_m:	?U
Gassignvariableop_76_adam_simple_model2_prediction_block2_dense_3_bias_m:\
Iassignvariableop_77_adam_simple_model2_prediction_block2_dense_4_kernel_m:	?U
Gassignvariableop_78_adam_simple_model2_prediction_block2_dense_4_bias_m:\
Iassignvariableop_79_adam_simple_model2_prediction_block2_dense_5_kernel_m:	?U
Gassignvariableop_80_adam_simple_model2_prediction_block2_dense_5_bias_m:\
Iassignvariableop_81_adam_simple_model2_prediction_block2_dense_6_kernel_m:	?U
Gassignvariableop_82_adam_simple_model2_prediction_block2_dense_6_bias_m:[
Aassignvariableop_83_adam_simple_model2_conv_block_conv2d_kernel_v: M
?assignvariableop_84_adam_simple_model2_conv_block_conv2d_bias_v: ]
Cassignvariableop_85_adam_simple_model2_conv_block_conv2d_1_kernel_v:  O
Aassignvariableop_86_adam_simple_model2_conv_block_conv2d_1_bias_v: _
Eassignvariableop_87_adam_simple_model2_conv_block_1_conv2d_2_kernel_v: @Q
Cassignvariableop_88_adam_simple_model2_conv_block_1_conv2d_2_bias_v:@_
Eassignvariableop_89_adam_simple_model2_conv_block_1_conv2d_3_kernel_v:@@Q
Cassignvariableop_90_adam_simple_model2_conv_block_1_conv2d_3_bias_v:@e
Jassignvariableop_91_adam_simple_model2_larger_conv_block_conv2d_4_kernel_v:@?W
Hassignvariableop_92_adam_simple_model2_larger_conv_block_conv2d_4_bias_v:	?f
Jassignvariableop_93_adam_simple_model2_larger_conv_block_conv2d_5_kernel_v:??W
Hassignvariableop_94_adam_simple_model2_larger_conv_block_conv2d_5_bias_v:	?f
Jassignvariableop_95_adam_simple_model2_larger_conv_block_conv2d_6_kernel_v:??W
Hassignvariableop_96_adam_simple_model2_larger_conv_block_conv2d_6_bias_v:	?[
Gassignvariableop_97_adam_simple_model2_prediction_block2_dense_kernel_v:
??T
Eassignvariableop_98_adam_simple_model2_prediction_block2_dense_bias_v:	?]
Iassignvariableop_99_adam_simple_model2_prediction_block2_dense_1_kernel_v:
??W
Hassignvariableop_100_adam_simple_model2_prediction_block2_dense_1_bias_v:	?]
Jassignvariableop_101_adam_simple_model2_prediction_block2_dense_2_kernel_v:	?V
Hassignvariableop_102_adam_simple_model2_prediction_block2_dense_2_bias_v:]
Jassignvariableop_103_adam_simple_model2_prediction_block2_dense_3_kernel_v:	?V
Hassignvariableop_104_adam_simple_model2_prediction_block2_dense_3_bias_v:]
Jassignvariableop_105_adam_simple_model2_prediction_block2_dense_4_kernel_v:	?V
Hassignvariableop_106_adam_simple_model2_prediction_block2_dense_4_bias_v:]
Jassignvariableop_107_adam_simple_model2_prediction_block2_dense_5_kernel_v:	?V
Hassignvariableop_108_adam_simple_model2_prediction_block2_dense_5_bias_v:]
Jassignvariableop_109_adam_simple_model2_prediction_block2_dense_6_kernel_v:	?V
Hassignvariableop_110_adam_simple_model2_prediction_block2_dense_6_bias_v:
identity_112??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?2
value?2B?2pB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:p*
dtype0*?
value?B?pB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*~
dtypest
r2p	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_simple_model2_conv_block_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp7assignvariableop_6_simple_model2_conv_block_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp;assignvariableop_7_simple_model2_conv_block_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_simple_model2_conv_block_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp=assignvariableop_9_simple_model2_conv_block_1_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp<assignvariableop_10_simple_model2_conv_block_1_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp>assignvariableop_11_simple_model2_conv_block_1_conv2d_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp<assignvariableop_12_simple_model2_conv_block_1_conv2d_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpCassignvariableop_13_simple_model2_larger_conv_block_conv2d_4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpAassignvariableop_14_simple_model2_larger_conv_block_conv2d_4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpCassignvariableop_15_simple_model2_larger_conv_block_conv2d_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpAassignvariableop_16_simple_model2_larger_conv_block_conv2d_5_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpCassignvariableop_17_simple_model2_larger_conv_block_conv2d_6_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpAassignvariableop_18_simple_model2_larger_conv_block_conv2d_6_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp@assignvariableop_19_simple_model2_prediction_block2_dense_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp>assignvariableop_20_simple_model2_prediction_block2_dense_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpBassignvariableop_21_simple_model2_prediction_block2_dense_1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_simple_model2_prediction_block2_dense_1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpBassignvariableop_23_simple_model2_prediction_block2_dense_2_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp@assignvariableop_24_simple_model2_prediction_block2_dense_2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpBassignvariableop_25_simple_model2_prediction_block2_dense_3_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp@assignvariableop_26_simple_model2_prediction_block2_dense_3_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpBassignvariableop_27_simple_model2_prediction_block2_dense_4_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp@assignvariableop_28_simple_model2_prediction_block2_dense_4_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpBassignvariableop_29_simple_model2_prediction_block2_dense_5_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_simple_model2_prediction_block2_dense_5_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpBassignvariableop_31_simple_model2_prediction_block2_dense_6_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp@assignvariableop_32_simple_model2_prediction_block2_dense_6_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_3Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_3Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_4Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_4Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_5Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_5Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_6Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_6Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_7Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_7Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_8Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_8Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_9Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_9Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_10Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_10Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpAassignvariableop_55_adam_simple_model2_conv_block_conv2d_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp?assignvariableop_56_adam_simple_model2_conv_block_conv2d_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpCassignvariableop_57_adam_simple_model2_conv_block_conv2d_1_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpAassignvariableop_58_adam_simple_model2_conv_block_conv2d_1_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpEassignvariableop_59_adam_simple_model2_conv_block_1_conv2d_2_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpCassignvariableop_60_adam_simple_model2_conv_block_1_conv2d_2_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpEassignvariableop_61_adam_simple_model2_conv_block_1_conv2d_3_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpCassignvariableop_62_adam_simple_model2_conv_block_1_conv2d_3_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpJassignvariableop_63_adam_simple_model2_larger_conv_block_conv2d_4_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpHassignvariableop_64_adam_simple_model2_larger_conv_block_conv2d_4_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpJassignvariableop_65_adam_simple_model2_larger_conv_block_conv2d_5_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpHassignvariableop_66_adam_simple_model2_larger_conv_block_conv2d_5_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpJassignvariableop_67_adam_simple_model2_larger_conv_block_conv2d_6_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpHassignvariableop_68_adam_simple_model2_larger_conv_block_conv2d_6_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpGassignvariableop_69_adam_simple_model2_prediction_block2_dense_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpEassignvariableop_70_adam_simple_model2_prediction_block2_dense_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpIassignvariableop_71_adam_simple_model2_prediction_block2_dense_1_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpGassignvariableop_72_adam_simple_model2_prediction_block2_dense_1_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOpIassignvariableop_73_adam_simple_model2_prediction_block2_dense_2_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpGassignvariableop_74_adam_simple_model2_prediction_block2_dense_2_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpIassignvariableop_75_adam_simple_model2_prediction_block2_dense_3_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOpGassignvariableop_76_adam_simple_model2_prediction_block2_dense_3_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOpIassignvariableop_77_adam_simple_model2_prediction_block2_dense_4_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOpGassignvariableop_78_adam_simple_model2_prediction_block2_dense_4_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOpIassignvariableop_79_adam_simple_model2_prediction_block2_dense_5_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOpGassignvariableop_80_adam_simple_model2_prediction_block2_dense_5_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOpIassignvariableop_81_adam_simple_model2_prediction_block2_dense_6_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpGassignvariableop_82_adam_simple_model2_prediction_block2_dense_6_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpAassignvariableop_83_adam_simple_model2_conv_block_conv2d_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp?assignvariableop_84_adam_simple_model2_conv_block_conv2d_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOpCassignvariableop_85_adam_simple_model2_conv_block_conv2d_1_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOpAassignvariableop_86_adam_simple_model2_conv_block_conv2d_1_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpEassignvariableop_87_adam_simple_model2_conv_block_1_conv2d_2_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpCassignvariableop_88_adam_simple_model2_conv_block_1_conv2d_2_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOpEassignvariableop_89_adam_simple_model2_conv_block_1_conv2d_3_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpCassignvariableop_90_adam_simple_model2_conv_block_1_conv2d_3_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOpJassignvariableop_91_adam_simple_model2_larger_conv_block_conv2d_4_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOpHassignvariableop_92_adam_simple_model2_larger_conv_block_conv2d_4_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOpJassignvariableop_93_adam_simple_model2_larger_conv_block_conv2d_5_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOpHassignvariableop_94_adam_simple_model2_larger_conv_block_conv2d_5_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpJassignvariableop_95_adam_simple_model2_larger_conv_block_conv2d_6_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpHassignvariableop_96_adam_simple_model2_larger_conv_block_conv2d_6_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOpGassignvariableop_97_adam_simple_model2_prediction_block2_dense_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOpEassignvariableop_98_adam_simple_model2_prediction_block2_dense_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOpIassignvariableop_99_adam_simple_model2_prediction_block2_dense_1_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOpHassignvariableop_100_adam_simple_model2_prediction_block2_dense_1_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOpJassignvariableop_101_adam_simple_model2_prediction_block2_dense_2_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOpHassignvariableop_102_adam_simple_model2_prediction_block2_dense_2_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOpJassignvariableop_103_adam_simple_model2_prediction_block2_dense_3_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOpHassignvariableop_104_adam_simple_model2_prediction_block2_dense_3_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOpJassignvariableop_105_adam_simple_model2_prediction_block2_dense_4_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOpHassignvariableop_106_adam_simple_model2_prediction_block2_dense_4_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOpJassignvariableop_107_adam_simple_model2_prediction_block2_dense_5_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOpHassignvariableop_108_adam_simple_model2_prediction_block2_dense_5_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOpJassignvariableop_109_adam_simple_model2_prediction_block2_dense_6_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOpHassignvariableop_110_adam_simple_model2_prediction_block2_dense_6_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_111Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_111?
Identity_112IdentityIdentity_111:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_112"%
identity_112Identity_112:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
H__inference_conv_block_1_layer_call_and_return_conditional_losses_125818

inputsA
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_2/LeakyRelu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_3/LeakyRelu?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
dropout_2/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_2/Identity?
IdentityIdentitydropout_2/Identity:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
ӥ
?
I__inference_simple_model2_layer_call_and_return_conditional_losses_127232

inputsJ
0conv_block_conv2d_conv2d_readvariableop_resource: ?
1conv_block_conv2d_biasadd_readvariableop_resource: L
2conv_block_conv2d_1_conv2d_readvariableop_resource:  A
3conv_block_conv2d_1_biasadd_readvariableop_resource: N
4conv_block_1_conv2d_2_conv2d_readvariableop_resource: @C
5conv_block_1_conv2d_2_biasadd_readvariableop_resource:@N
4conv_block_1_conv2d_3_conv2d_readvariableop_resource:@@C
5conv_block_1_conv2d_3_biasadd_readvariableop_resource:@T
9larger_conv_block_conv2d_4_conv2d_readvariableop_resource:@?I
:larger_conv_block_conv2d_4_biasadd_readvariableop_resource:	?U
9larger_conv_block_conv2d_5_conv2d_readvariableop_resource:??I
:larger_conv_block_conv2d_5_biasadd_readvariableop_resource:	?U
9larger_conv_block_conv2d_6_conv2d_readvariableop_resource:??I
:larger_conv_block_conv2d_6_biasadd_readvariableop_resource:	?J
6prediction_block2_dense_matmul_readvariableop_resource:
??F
7prediction_block2_dense_biasadd_readvariableop_resource:	?L
8prediction_block2_dense_1_matmul_readvariableop_resource:
??H
9prediction_block2_dense_1_biasadd_readvariableop_resource:	?K
8prediction_block2_dense_2_matmul_readvariableop_resource:	?G
9prediction_block2_dense_2_biasadd_readvariableop_resource:K
8prediction_block2_dense_3_matmul_readvariableop_resource:	?G
9prediction_block2_dense_3_biasadd_readvariableop_resource:K
8prediction_block2_dense_4_matmul_readvariableop_resource:	?G
9prediction_block2_dense_4_biasadd_readvariableop_resource:K
8prediction_block2_dense_5_matmul_readvariableop_resource:	?G
9prediction_block2_dense_5_biasadd_readvariableop_resource:K
8prediction_block2_dense_6_matmul_readvariableop_resource:	?G
9prediction_block2_dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4??(conv_block/conv2d/BiasAdd/ReadVariableOp?'conv_block/conv2d/Conv2D/ReadVariableOp?*conv_block/conv2d_1/BiasAdd/ReadVariableOp?)conv_block/conv2d_1/Conv2D/ReadVariableOp?,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp?+conv_block_1/conv2d_2/Conv2D/ReadVariableOp?,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp?+conv_block_1/conv2d_3/Conv2D/ReadVariableOp?1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp?1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp?1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp?.prediction_block2/dense/BiasAdd/ReadVariableOp?-prediction_block2/dense/MatMul/ReadVariableOp?0prediction_block2/dense_1/BiasAdd/ReadVariableOp?/prediction_block2/dense_1/MatMul/ReadVariableOp?0prediction_block2/dense_2/BiasAdd/ReadVariableOp?/prediction_block2/dense_2/MatMul/ReadVariableOp?0prediction_block2/dense_3/BiasAdd/ReadVariableOp?/prediction_block2/dense_3/MatMul/ReadVariableOp?0prediction_block2/dense_4/BiasAdd/ReadVariableOp?/prediction_block2/dense_4/MatMul/ReadVariableOp?0prediction_block2/dense_5/BiasAdd/ReadVariableOp?/prediction_block2/dense_5/MatMul/ReadVariableOp?0prediction_block2/dense_6/BiasAdd/ReadVariableOp?/prediction_block2/dense_6/MatMul/ReadVariableOps
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMulinputsdropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@@2
dropout/dropout/Muld
dropout/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@@2
dropout/dropout/Mul_1?
'conv_block/conv2d/Conv2D/ReadVariableOpReadVariableOp0conv_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'conv_block/conv2d/Conv2D/ReadVariableOp?
conv_block/conv2d/Conv2DConv2Ddropout/dropout/Mul_1:z:0/conv_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingVALID*
strides
2
conv_block/conv2d/Conv2D?
(conv_block/conv2d/BiasAdd/ReadVariableOpReadVariableOp1conv_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(conv_block/conv2d/BiasAdd/ReadVariableOp?
conv_block/conv2d/BiasAddBiasAdd!conv_block/conv2d/Conv2D:output:00conv_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv_block/conv2d/BiasAdd?
 conv_block/leaky_re_lu/LeakyRelu	LeakyRelu"conv_block/conv2d/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2"
 conv_block/leaky_re_lu/LeakyRelu?
)conv_block/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2conv_block_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02+
)conv_block/conv2d_1/Conv2D/ReadVariableOp?
conv_block/conv2d_1/Conv2DConv2D.conv_block/leaky_re_lu/LeakyRelu:activations:01conv_block/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingSAME*
strides
2
conv_block/conv2d_1/Conv2D?
*conv_block/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3conv_block_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv_block/conv2d_1/BiasAdd/ReadVariableOp?
conv_block/conv2d_1/BiasAddBiasAdd#conv_block/conv2d_1/Conv2D:output:02conv_block/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv_block/conv2d_1/BiasAdd?
"conv_block/leaky_re_lu_1/LeakyRelu	LeakyRelu$conv_block/conv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2$
"conv_block/leaky_re_lu_1/LeakyRelu?
 conv_block/max_pooling2d/MaxPoolMaxPool0conv_block/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2"
 conv_block/max_pooling2d/MaxPool?
"conv_block/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2$
"conv_block/dropout_1/dropout/Const?
 conv_block/dropout_1/dropout/MulMul)conv_block/max_pooling2d/MaxPool:output:0+conv_block/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2"
 conv_block/dropout_1/dropout/Mul?
"conv_block/dropout_1/dropout/ShapeShape)conv_block/max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2$
"conv_block/dropout_1/dropout/Shape?
9conv_block/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+conv_block/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02;
9conv_block/dropout_1/dropout/random_uniform/RandomUniform?
+conv_block/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2-
+conv_block/dropout_1/dropout/GreaterEqual/y?
)conv_block/dropout_1/dropout/GreaterEqualGreaterEqualBconv_block/dropout_1/dropout/random_uniform/RandomUniform:output:04conv_block/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2+
)conv_block/dropout_1/dropout/GreaterEqual?
!conv_block/dropout_1/dropout/CastCast-conv_block/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2#
!conv_block/dropout_1/dropout/Cast?
"conv_block/dropout_1/dropout/Mul_1Mul$conv_block/dropout_1/dropout/Mul:z:0%conv_block/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2$
"conv_block/dropout_1/dropout/Mul_1?
+conv_block_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4conv_block_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+conv_block_1/conv2d_2/Conv2D/ReadVariableOp?
conv_block_1/conv2d_2/Conv2DConv2D&conv_block/dropout_1/dropout/Mul_1:z:03conv_block_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv_block_1/conv2d_2/Conv2D?
,conv_block_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5conv_block_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp?
conv_block_1/conv2d_2/BiasAddBiasAdd%conv_block_1/conv2d_2/Conv2D:output:04conv_block_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv_block_1/conv2d_2/BiasAdd?
$conv_block_1/leaky_re_lu_2/LeakyRelu	LeakyRelu&conv_block_1/conv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2&
$conv_block_1/leaky_re_lu_2/LeakyRelu?
+conv_block_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4conv_block_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+conv_block_1/conv2d_3/Conv2D/ReadVariableOp?
conv_block_1/conv2d_3/Conv2DConv2D2conv_block_1/leaky_re_lu_2/LeakyRelu:activations:03conv_block_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv_block_1/conv2d_3/Conv2D?
,conv_block_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5conv_block_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp?
conv_block_1/conv2d_3/BiasAddBiasAdd%conv_block_1/conv2d_3/Conv2D:output:04conv_block_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv_block_1/conv2d_3/BiasAdd?
$conv_block_1/leaky_re_lu_3/LeakyRelu	LeakyRelu&conv_block_1/conv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2&
$conv_block_1/leaky_re_lu_3/LeakyRelu?
$conv_block_1/max_pooling2d_1/MaxPoolMaxPool2conv_block_1/leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$conv_block_1/max_pooling2d_1/MaxPool?
$conv_block_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2&
$conv_block_1/dropout_2/dropout/Const?
"conv_block_1/dropout_2/dropout/MulMul-conv_block_1/max_pooling2d_1/MaxPool:output:0-conv_block_1/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2$
"conv_block_1/dropout_2/dropout/Mul?
$conv_block_1/dropout_2/dropout/ShapeShape-conv_block_1/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2&
$conv_block_1/dropout_2/dropout/Shape?
;conv_block_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform-conv_block_1/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02=
;conv_block_1/dropout_2/dropout/random_uniform/RandomUniform?
-conv_block_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2/
-conv_block_1/dropout_2/dropout/GreaterEqual/y?
+conv_block_1/dropout_2/dropout/GreaterEqualGreaterEqualDconv_block_1/dropout_2/dropout/random_uniform/RandomUniform:output:06conv_block_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2-
+conv_block_1/dropout_2/dropout/GreaterEqual?
#conv_block_1/dropout_2/dropout/CastCast/conv_block_1/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2%
#conv_block_1/dropout_2/dropout/Cast?
$conv_block_1/dropout_2/dropout/Mul_1Mul&conv_block_1/dropout_2/dropout/Mul:z:0'conv_block_1/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2&
$conv_block_1/dropout_2/dropout/Mul_1?
0larger_conv_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp9larger_conv_block_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype022
0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp?
!larger_conv_block/conv2d_4/Conv2DConv2D(conv_block_1/dropout_2/dropout/Mul_1:z:08larger_conv_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2#
!larger_conv_block/conv2d_4/Conv2D?
1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp:larger_conv_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp?
"larger_conv_block/conv2d_4/BiasAddBiasAdd*larger_conv_block/conv2d_4/Conv2D:output:09larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2$
"larger_conv_block/conv2d_4/BiasAdd?
)larger_conv_block/leaky_re_lu_4/LeakyRelu	LeakyRelu+larger_conv_block/conv2d_4/BiasAdd:output:0*0
_output_shapes
:?????????		?*
alpha%???>2+
)larger_conv_block/leaky_re_lu_4/LeakyRelu?
0larger_conv_block/conv2d_5/Conv2D/ReadVariableOpReadVariableOp9larger_conv_block_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp?
!larger_conv_block/conv2d_5/Conv2DConv2D7larger_conv_block/leaky_re_lu_4/LeakyRelu:activations:08larger_conv_block/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2#
!larger_conv_block/conv2d_5/Conv2D?
1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp:larger_conv_block_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp?
"larger_conv_block/conv2d_5/BiasAddBiasAdd*larger_conv_block/conv2d_5/Conv2D:output:09larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"larger_conv_block/conv2d_5/BiasAdd?
)larger_conv_block/leaky_re_lu_5/LeakyRelu	LeakyRelu+larger_conv_block/conv2d_5/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2+
)larger_conv_block/leaky_re_lu_5/LeakyRelu?
0larger_conv_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp9larger_conv_block_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp?
!larger_conv_block/conv2d_6/Conv2DConv2D7larger_conv_block/leaky_re_lu_5/LeakyRelu:activations:08larger_conv_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!larger_conv_block/conv2d_6/Conv2D?
1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp:larger_conv_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp?
"larger_conv_block/conv2d_6/BiasAddBiasAdd*larger_conv_block/conv2d_6/Conv2D:output:09larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2$
"larger_conv_block/conv2d_6/BiasAdd?
)larger_conv_block/leaky_re_lu_6/LeakyRelu	LeakyRelu+larger_conv_block/conv2d_6/BiasAdd:output:0*0
_output_shapes
:??????????*
alpha%???>2+
)larger_conv_block/leaky_re_lu_6/LeakyRelu?
)larger_conv_block/max_pooling2d_2/MaxPoolMaxPool7larger_conv_block/leaky_re_lu_6/LeakyRelu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2+
)larger_conv_block/max_pooling2d_2/MaxPool?
prediction_block2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2!
prediction_block2/flatten/Const?
!prediction_block2/flatten/ReshapeReshape2larger_conv_block/max_pooling2d_2/MaxPool:output:0(prediction_block2/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2#
!prediction_block2/flatten/Reshape?
-prediction_block2/dense/MatMul/ReadVariableOpReadVariableOp6prediction_block2_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-prediction_block2/dense/MatMul/ReadVariableOp?
prediction_block2/dense/MatMulMatMul*prediction_block2/flatten/Reshape:output:05prediction_block2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
prediction_block2/dense/MatMul?
.prediction_block2/dense/BiasAdd/ReadVariableOpReadVariableOp7prediction_block2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.prediction_block2/dense/BiasAdd/ReadVariableOp?
prediction_block2/dense/BiasAddBiasAdd(prediction_block2/dense/MatMul:product:06prediction_block2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
prediction_block2/dense/BiasAdd?
"prediction_block2/dense/re_lu/ReluRelu(prediction_block2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2$
"prediction_block2/dense/re_lu/Relu?
)prediction_block2/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)prediction_block2/dropout_3/dropout/Const?
'prediction_block2/dropout_3/dropout/MulMul0prediction_block2/dense/re_lu/Relu:activations:02prediction_block2/dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2)
'prediction_block2/dropout_3/dropout/Mul?
)prediction_block2/dropout_3/dropout/ShapeShape0prediction_block2/dense/re_lu/Relu:activations:0*
T0*
_output_shapes
:2+
)prediction_block2/dropout_3/dropout/Shape?
@prediction_block2/dropout_3/dropout/random_uniform/RandomUniformRandomUniform2prediction_block2/dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02B
@prediction_block2/dropout_3/dropout/random_uniform/RandomUniform?
2prediction_block2/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2prediction_block2/dropout_3/dropout/GreaterEqual/y?
0prediction_block2/dropout_3/dropout/GreaterEqualGreaterEqualIprediction_block2/dropout_3/dropout/random_uniform/RandomUniform:output:0;prediction_block2/dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????22
0prediction_block2/dropout_3/dropout/GreaterEqual?
(prediction_block2/dropout_3/dropout/CastCast4prediction_block2/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2*
(prediction_block2/dropout_3/dropout/Cast?
)prediction_block2/dropout_3/dropout/Mul_1Mul+prediction_block2/dropout_3/dropout/Mul:z:0,prediction_block2/dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2+
)prediction_block2/dropout_3/dropout/Mul_1?
/prediction_block2/dense_1/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/prediction_block2/dense_1/MatMul/ReadVariableOp?
 prediction_block2/dense_1/MatMulMatMul-prediction_block2/dropout_3/dropout/Mul_1:z:07prediction_block2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 prediction_block2/dense_1/MatMul?
0prediction_block2/dense_1/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0prediction_block2/dense_1/BiasAdd/ReadVariableOp?
!prediction_block2/dense_1/BiasAddBiasAdd*prediction_block2/dense_1/MatMul:product:08prediction_block2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!prediction_block2/dense_1/BiasAdd?
&prediction_block2/dense_1/re_lu_1/ReluRelu*prediction_block2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&prediction_block2/dense_1/re_lu_1/Relu?
/prediction_block2/dense_2/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_2/MatMul/ReadVariableOp?
 prediction_block2/dense_2/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_2/MatMul?
0prediction_block2/dense_2/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_2/BiasAdd/ReadVariableOp?
!prediction_block2/dense_2/BiasAddBiasAdd*prediction_block2/dense_2/MatMul:product:08prediction_block2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_2/BiasAdd?
!prediction_block2/dense_2/SoftmaxSoftmax*prediction_block2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_2/Softmax?
/prediction_block2/dense_3/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_3/MatMul/ReadVariableOp?
 prediction_block2/dense_3/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_3/MatMul?
0prediction_block2/dense_3/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_3/BiasAdd/ReadVariableOp?
!prediction_block2/dense_3/BiasAddBiasAdd*prediction_block2/dense_3/MatMul:product:08prediction_block2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_3/BiasAdd?
!prediction_block2/dense_3/SoftmaxSoftmax*prediction_block2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_3/Softmax?
/prediction_block2/dense_4/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_4/MatMul/ReadVariableOp?
 prediction_block2/dense_4/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_4/MatMul?
0prediction_block2/dense_4/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_4/BiasAdd/ReadVariableOp?
!prediction_block2/dense_4/BiasAddBiasAdd*prediction_block2/dense_4/MatMul:product:08prediction_block2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_4/BiasAdd?
!prediction_block2/dense_4/SoftmaxSoftmax*prediction_block2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_4/Softmax?
/prediction_block2/dense_5/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_5/MatMul/ReadVariableOp?
 prediction_block2/dense_5/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_5/MatMul?
0prediction_block2/dense_5/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_5/BiasAdd/ReadVariableOp?
!prediction_block2/dense_5/BiasAddBiasAdd*prediction_block2/dense_5/MatMul:product:08prediction_block2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_5/BiasAdd?
!prediction_block2/dense_5/SoftmaxSoftmax*prediction_block2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_5/Softmax?
/prediction_block2/dense_6/MatMul/ReadVariableOpReadVariableOp8prediction_block2_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/prediction_block2/dense_6/MatMul/ReadVariableOp?
 prediction_block2/dense_6/MatMulMatMul4prediction_block2/dense_1/re_lu_1/Relu:activations:07prediction_block2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prediction_block2/dense_6/MatMul?
0prediction_block2/dense_6/BiasAdd/ReadVariableOpReadVariableOp9prediction_block2_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prediction_block2/dense_6/BiasAdd/ReadVariableOp?
!prediction_block2/dense_6/BiasAddBiasAdd*prediction_block2/dense_6/MatMul:product:08prediction_block2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_6/BiasAdd?
!prediction_block2/dense_6/SoftmaxSoftmax*prediction_block2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!prediction_block2/dense_6/Softmax?
IdentityIdentity+prediction_block2/dense_2/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity+prediction_block2/dense_3/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+prediction_block2/dense_4/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity+prediction_block2/dense_5/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity+prediction_block2/dense_6/Softmax:softmax:0)^conv_block/conv2d/BiasAdd/ReadVariableOp(^conv_block/conv2d/Conv2D/ReadVariableOp+^conv_block/conv2d_1/BiasAdd/ReadVariableOp*^conv_block/conv2d_1/Conv2D/ReadVariableOp-^conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_2/Conv2D/ReadVariableOp-^conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,^conv_block_1/conv2d_3/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2^larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1^larger_conv_block/conv2d_6/Conv2D/ReadVariableOp/^prediction_block2/dense/BiasAdd/ReadVariableOp.^prediction_block2/dense/MatMul/ReadVariableOp1^prediction_block2/dense_1/BiasAdd/ReadVariableOp0^prediction_block2/dense_1/MatMul/ReadVariableOp1^prediction_block2/dense_2/BiasAdd/ReadVariableOp0^prediction_block2/dense_2/MatMul/ReadVariableOp1^prediction_block2/dense_3/BiasAdd/ReadVariableOp0^prediction_block2/dense_3/MatMul/ReadVariableOp1^prediction_block2/dense_4/BiasAdd/ReadVariableOp0^prediction_block2/dense_4/MatMul/ReadVariableOp1^prediction_block2/dense_5/BiasAdd/ReadVariableOp0^prediction_block2/dense_5/MatMul/ReadVariableOp1^prediction_block2/dense_6/BiasAdd/ReadVariableOp0^prediction_block2/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(conv_block/conv2d/BiasAdd/ReadVariableOp(conv_block/conv2d/BiasAdd/ReadVariableOp2R
'conv_block/conv2d/Conv2D/ReadVariableOp'conv_block/conv2d/Conv2D/ReadVariableOp2X
*conv_block/conv2d_1/BiasAdd/ReadVariableOp*conv_block/conv2d_1/BiasAdd/ReadVariableOp2V
)conv_block/conv2d_1/Conv2D/ReadVariableOp)conv_block/conv2d_1/Conv2D/ReadVariableOp2\
,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp,conv_block_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+conv_block_1/conv2d_2/Conv2D/ReadVariableOp+conv_block_1/conv2d_2/Conv2D/ReadVariableOp2\
,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp,conv_block_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+conv_block_1/conv2d_3/Conv2D/ReadVariableOp+conv_block_1/conv2d_3/Conv2D/ReadVariableOp2f
1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp1larger_conv_block/conv2d_4/BiasAdd/ReadVariableOp2d
0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp0larger_conv_block/conv2d_4/Conv2D/ReadVariableOp2f
1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp1larger_conv_block/conv2d_5/BiasAdd/ReadVariableOp2d
0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp0larger_conv_block/conv2d_5/Conv2D/ReadVariableOp2f
1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp1larger_conv_block/conv2d_6/BiasAdd/ReadVariableOp2d
0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp0larger_conv_block/conv2d_6/Conv2D/ReadVariableOp2`
.prediction_block2/dense/BiasAdd/ReadVariableOp.prediction_block2/dense/BiasAdd/ReadVariableOp2^
-prediction_block2/dense/MatMul/ReadVariableOp-prediction_block2/dense/MatMul/ReadVariableOp2d
0prediction_block2/dense_1/BiasAdd/ReadVariableOp0prediction_block2/dense_1/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_1/MatMul/ReadVariableOp/prediction_block2/dense_1/MatMul/ReadVariableOp2d
0prediction_block2/dense_2/BiasAdd/ReadVariableOp0prediction_block2/dense_2/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_2/MatMul/ReadVariableOp/prediction_block2/dense_2/MatMul/ReadVariableOp2d
0prediction_block2/dense_3/BiasAdd/ReadVariableOp0prediction_block2/dense_3/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_3/MatMul/ReadVariableOp/prediction_block2/dense_3/MatMul/ReadVariableOp2d
0prediction_block2/dense_4/BiasAdd/ReadVariableOp0prediction_block2/dense_4/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_4/MatMul/ReadVariableOp/prediction_block2/dense_4/MatMul/ReadVariableOp2d
0prediction_block2/dense_5/BiasAdd/ReadVariableOp0prediction_block2/dense_5/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_5/MatMul/ReadVariableOp/prediction_block2/dense_5/MatMul/ReadVariableOp2d
0prediction_block2/dense_6/BiasAdd/ReadVariableOp0prediction_block2/dense_6/BiasAdd/ReadVariableOp2b
/prediction_block2/dense_6/MatMul/ReadVariableOp/prediction_block2/dense_6/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_127247

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
F__inference_conv_block_layer_call_and_return_conditional_losses_127305

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: 
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu/LeakyRelu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D#leaky_re_lu/LeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu_1/LeakyRelu?
max_pooling2d/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout_1/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
dropout_1/Identity?
IdentityIdentitydropout_1/Identity:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
F__inference_conv_block_layer_call_and_return_conditional_losses_125788

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: 
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu/LeakyRelu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D#leaky_re_lu/LeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<< 2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????<< *
alpha%???>2
leaky_re_lu_1/LeakyRelu?
max_pooling2d/MaxPoolMaxPool%leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout_1/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
dropout_1/Identity?
IdentityIdentitydropout_1/Identity:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_1_layer_call_fn_125742

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1257362
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_2_layer_call_fn_125754

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1257482
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_127242

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1263302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?4
?
I__inference_simple_model2_layer_call_and_return_conditional_losses_126686
input_1+
conv_block_126617: 
conv_block_126619: +
conv_block_126621:  
conv_block_126623: -
conv_block_1_126626: @!
conv_block_1_126628:@-
conv_block_1_126630:@@!
conv_block_1_126632:@3
larger_conv_block_126635:@?'
larger_conv_block_126637:	?4
larger_conv_block_126639:??'
larger_conv_block_126641:	?4
larger_conv_block_126643:??'
larger_conv_block_126645:	?,
prediction_block2_126648:
??'
prediction_block2_126650:	?,
prediction_block2_126652:
??'
prediction_block2_126654:	?+
prediction_block2_126656:	?&
prediction_block2_126658:+
prediction_block2_126660:	?&
prediction_block2_126662:+
prediction_block2_126664:	?&
prediction_block2_126666:+
prediction_block2_126668:	?&
prediction_block2_126670:+
prediction_block2_126672:	?&
prediction_block2_126674:
identity

identity_1

identity_2

identity_3

identity_4??"conv_block/StatefulPartitionedCall?$conv_block_1/StatefulPartitionedCall?)larger_conv_block/StatefulPartitionedCall?)prediction_block2/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1257662
dropout/PartitionedCall?
"conv_block/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv_block_126617conv_block_126619conv_block_126621conv_block_126623*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_block_layer_call_and_return_conditional_losses_1257882$
"conv_block/StatefulPartitionedCall?
$conv_block_1/StatefulPartitionedCallStatefulPartitionedCall+conv_block/StatefulPartitionedCall:output:0conv_block_1_126626conv_block_1_126628conv_block_1_126630conv_block_1_126632*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_1_layer_call_and_return_conditional_losses_1258182&
$conv_block_1/StatefulPartitionedCall?
)larger_conv_block/StatefulPartitionedCallStatefulPartitionedCall-conv_block_1/StatefulPartitionedCall:output:0larger_conv_block_126635larger_conv_block_126637larger_conv_block_126639larger_conv_block_126641larger_conv_block_126643larger_conv_block_126645*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_1258542+
)larger_conv_block/StatefulPartitionedCall?
)prediction_block2/StatefulPartitionedCallStatefulPartitionedCall2larger_conv_block/StatefulPartitionedCall:output:0prediction_block2_126648prediction_block2_126650prediction_block2_126652prediction_block2_126654prediction_block2_126656prediction_block2_126658prediction_block2_126660prediction_block2_126662prediction_block2_126664prediction_block2_126666prediction_block2_126668prediction_block2_126670prediction_block2_126672prediction_block2_126674*
Tin
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_prediction_block2_layer_call_and_return_conditional_losses_1259282+
)prediction_block2/StatefulPartitionedCall?
IdentityIdentity2prediction_block2/StatefulPartitionedCall:output:0#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity2prediction_block2/StatefulPartitionedCall:output:1#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity2prediction_block2/StatefulPartitionedCall:output:2#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity2prediction_block2/StatefulPartitionedCall:output:3#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity2prediction_block2/StatefulPartitionedCall:output:4#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"conv_block/StatefulPartitionedCall"conv_block/StatefulPartitionedCall2L
$conv_block_1/StatefulPartitionedCall$conv_block_1/StatefulPartitionedCall2V
)larger_conv_block/StatefulPartitionedCall)larger_conv_block/StatefulPartitionedCall2V
)prediction_block2/StatefulPartitionedCall)prediction_block2/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
J
.__inference_max_pooling2d_layer_call_fn_125730

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1257242
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_conv_block_1_layer_call_and_return_conditional_losses_127378

inputsA
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_2/LeakyRelu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/BiasAdd?
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_3/LeakyRelu?
max_pooling2d_1/MaxPoolMaxPool%leaky_re_lu_3/LeakyRelu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
dropout_2/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_2/Identity?
IdentityIdentitydropout_2/Identity:output:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?b
?
M__inference_prediction_block2_layer_call_and_return_conditional_losses_126145

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:9
&dense_6_matmul_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddw
dense/re_lu/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/re_lu/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const?
dropout_3/dropout/MulMuldense/re_lu/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeShapedense/re_lu/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAdd?
dense_1/re_lu_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/re_lu_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Softmax?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Softmax?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddy
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Softmax?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/Softmax?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul"dense_1/re_lu_1/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddy
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_6/Softmax?
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_3/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitydense_4/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identitydense_5/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identitydense_6/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_125748

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?4
?
I__inference_simple_model2_layer_call_and_return_conditional_losses_125967

inputs+
conv_block_125789: 
conv_block_125791: +
conv_block_125793:  
conv_block_125795: -
conv_block_1_125819: @!
conv_block_1_125821:@-
conv_block_1_125823:@@!
conv_block_1_125825:@3
larger_conv_block_125855:@?'
larger_conv_block_125857:	?4
larger_conv_block_125859:??'
larger_conv_block_125861:	?4
larger_conv_block_125863:??'
larger_conv_block_125865:	?,
prediction_block2_125929:
??'
prediction_block2_125931:	?,
prediction_block2_125933:
??'
prediction_block2_125935:	?+
prediction_block2_125937:	?&
prediction_block2_125939:+
prediction_block2_125941:	?&
prediction_block2_125943:+
prediction_block2_125945:	?&
prediction_block2_125947:+
prediction_block2_125949:	?&
prediction_block2_125951:+
prediction_block2_125953:	?&
prediction_block2_125955:
identity

identity_1

identity_2

identity_3

identity_4??"conv_block/StatefulPartitionedCall?$conv_block_1/StatefulPartitionedCall?)larger_conv_block/StatefulPartitionedCall?)prediction_block2/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1257662
dropout/PartitionedCall?
"conv_block/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv_block_125789conv_block_125791conv_block_125793conv_block_125795*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_block_layer_call_and_return_conditional_losses_1257882$
"conv_block/StatefulPartitionedCall?
$conv_block_1/StatefulPartitionedCallStatefulPartitionedCall+conv_block/StatefulPartitionedCall:output:0conv_block_1_125819conv_block_1_125821conv_block_1_125823conv_block_1_125825*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_1_layer_call_and_return_conditional_losses_1258182&
$conv_block_1/StatefulPartitionedCall?
)larger_conv_block/StatefulPartitionedCallStatefulPartitionedCall-conv_block_1/StatefulPartitionedCall:output:0larger_conv_block_125855larger_conv_block_125857larger_conv_block_125859larger_conv_block_125861larger_conv_block_125863larger_conv_block_125865*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_1258542+
)larger_conv_block/StatefulPartitionedCall?
)prediction_block2/StatefulPartitionedCallStatefulPartitionedCall2larger_conv_block/StatefulPartitionedCall:output:0prediction_block2_125929prediction_block2_125931prediction_block2_125933prediction_block2_125935prediction_block2_125937prediction_block2_125939prediction_block2_125941prediction_block2_125943prediction_block2_125945prediction_block2_125947prediction_block2_125949prediction_block2_125951prediction_block2_125953prediction_block2_125955*
Tin
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_prediction_block2_layer_call_and_return_conditional_losses_1259282+
)prediction_block2/StatefulPartitionedCall?
IdentityIdentity2prediction_block2/StatefulPartitionedCall:output:0#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity2prediction_block2/StatefulPartitionedCall:output:1#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity2prediction_block2/StatefulPartitionedCall:output:2#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity2prediction_block2/StatefulPartitionedCall:output:3#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity2prediction_block2/StatefulPartitionedCall:output:4#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"conv_block/StatefulPartitionedCall"conv_block/StatefulPartitionedCall2L
$conv_block_1/StatefulPartitionedCall$conv_block_1/StatefulPartitionedCall2V
)larger_conv_block/StatefulPartitionedCall)larger_conv_block/StatefulPartitionedCall2V
)prediction_block2/StatefulPartitionedCall)prediction_block2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
+__inference_conv_block_layer_call_fn_127285

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_block_layer_call_and_return_conditional_losses_1262992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
2__inference_prediction_block2_layer_call_fn_127530

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:
	unknown_9:	?

unknown_10:

unknown_11:	?

unknown_12:
identity

identity_1

identity_2

identity_3

identity_4??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_prediction_block2_layer_call_and_return_conditional_losses_1261452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?6
?
I__inference_simple_model2_layer_call_and_return_conditional_losses_126477

inputs+
conv_block_126408: 
conv_block_126410: +
conv_block_126412:  
conv_block_126414: -
conv_block_1_126417: @!
conv_block_1_126419:@-
conv_block_1_126421:@@!
conv_block_1_126423:@3
larger_conv_block_126426:@?'
larger_conv_block_126428:	?4
larger_conv_block_126430:??'
larger_conv_block_126432:	?4
larger_conv_block_126434:??'
larger_conv_block_126436:	?,
prediction_block2_126439:
??'
prediction_block2_126441:	?,
prediction_block2_126443:
??'
prediction_block2_126445:	?+
prediction_block2_126447:	?&
prediction_block2_126449:+
prediction_block2_126451:	?&
prediction_block2_126453:+
prediction_block2_126455:	?&
prediction_block2_126457:+
prediction_block2_126459:	?&
prediction_block2_126461:+
prediction_block2_126463:	?&
prediction_block2_126465:
identity

identity_1

identity_2

identity_3

identity_4??"conv_block/StatefulPartitionedCall?$conv_block_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?)larger_conv_block/StatefulPartitionedCall?)prediction_block2/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1263302!
dropout/StatefulPartitionedCall?
"conv_block/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv_block_126408conv_block_126410conv_block_126412conv_block_126414*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv_block_layer_call_and_return_conditional_losses_1262992$
"conv_block/StatefulPartitionedCall?
$conv_block_1/StatefulPartitionedCallStatefulPartitionedCall+conv_block/StatefulPartitionedCall:output:0conv_block_1_126417conv_block_1_126419conv_block_1_126421conv_block_1_126423*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_1_layer_call_and_return_conditional_losses_1262452&
$conv_block_1/StatefulPartitionedCall?
)larger_conv_block/StatefulPartitionedCallStatefulPartitionedCall-conv_block_1/StatefulPartitionedCall:output:0larger_conv_block_126426larger_conv_block_126428larger_conv_block_126430larger_conv_block_126432larger_conv_block_126434larger_conv_block_126436*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_1258542+
)larger_conv_block/StatefulPartitionedCall?
)prediction_block2/StatefulPartitionedCallStatefulPartitionedCall2larger_conv_block/StatefulPartitionedCall:output:0prediction_block2_126439prediction_block2_126441prediction_block2_126443prediction_block2_126445prediction_block2_126447prediction_block2_126449prediction_block2_126451prediction_block2_126453prediction_block2_126455prediction_block2_126457prediction_block2_126459prediction_block2_126461prediction_block2_126463prediction_block2_126465*
Tin
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_prediction_block2_layer_call_and_return_conditional_losses_1261452+
)prediction_block2/StatefulPartitionedCall?
IdentityIdentity2prediction_block2/StatefulPartitionedCall:output:0#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity2prediction_block2/StatefulPartitionedCall:output:1#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity2prediction_block2/StatefulPartitionedCall:output:2#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity2prediction_block2/StatefulPartitionedCall:output:3#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity2prediction_block2/StatefulPartitionedCall:output:4#^conv_block/StatefulPartitionedCall%^conv_block_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^larger_conv_block/StatefulPartitionedCall*^prediction_block2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"conv_block/StatefulPartitionedCall"conv_block/StatefulPartitionedCall2L
$conv_block_1/StatefulPartitionedCall$conv_block_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2V
)larger_conv_block/StatefulPartitionedCall)larger_conv_block/StatefulPartitionedCall2V
)prediction_block2/StatefulPartitionedCall)prediction_block2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
.__inference_simple_model2_layer_call_fn_126974

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:
??

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?

unknown_20:

unknown_21:	?

unknown_22:

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:
identity

identity_1

identity_2

identity_3

identity_4??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 *s
_output_shapesa
_:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_simple_model2_layer_call_and_return_conditional_losses_1264772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@@<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????<
output_30
StatefulPartitionedCall:2?????????<
output_40
StatefulPartitionedCall:3?????????<
output_50
StatefulPartitionedCall:4?????????tensorflow/serving/predict:ȷ
?
layer_1
block_1
block_2
block_3

prediction
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"name": "simple_model2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "SimpleModel2", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 64, 64, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "SimpleModel2"}, "training_config": {"loss": "CategoricalCrossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "output_1_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 0}], [{"class_name": "MeanMetricWrapper", "config": {"name": "output_2_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}], [{"class_name": "MeanMetricWrapper", "config": {"name": "output_3_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 2}], [{"class_name": "MeanMetricWrapper", "config": {"name": "output_4_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 3}], [{"class_name": "MeanMetricWrapper", "config": {"name": "output_5_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 4}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 5}
?
layer_1
layer_2
layer_3
layer_4
layer_5
layer_6
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "conv_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvBlock", "config": {"layer was saved without config": true}}
?
layer_1
layer_2
layer_3
layer_4
layer_5
layer_6
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "conv_block_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvBlock", "config": {"layer was saved without config": true}}
?
$layer_1
%layer_2
&layer_3
'layer_4
(layer_5
)layer_6
*layer_7
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "larger_conv_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LargerConvBlock", "config": {"layer was saved without config": true}}
?
/layer_1
0layer_2
1layer_3
2layer_4
3layer_5
4layer_6
5layer_7
6layer_8
7layer_9
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "prediction_block2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PredictionBlock2", "config": {"layer was saved without config": true}}
?
<iter

=beta_1

>beta_2
	?decay
@learning_rateAm?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?"
	optimizer
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17
S18
T19
U20
V21
W22
X23
Y24
Z25
[26
\27"
trackable_list_wrapper
?
A0
B1
C2
D3
E4
F5
G6
H7
I8
J9
K10
L11
M12
N13
O14
P15
Q16
R17
S18
T19
U20
V21
W22
X23
Y24
Z25
[26
\27"
trackable_list_wrapper
 "
trackable_list_wrapper
?
]layer_regularization_losses
^non_trainable_variables
	variables
trainable_variables
	regularization_losses
_layer_metrics
`metrics

alayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
blayer_regularization_losses
cnon_trainable_variables
	variables
trainable_variables
regularization_losses
dlayer_metrics
emetrics

flayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


Akernel
Bbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 9}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 1]}}
?
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 10}
?


Ckernel
Dbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 60, 32]}}
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 15}
?
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 17}}
?
{	variables
|trainable_variables
}regularization_losses
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 18}
<
A0
B1
C2
D3"
trackable_list_wrapper
<
A0
B1
C2
D3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_regularization_losses
?non_trainable_variables
	variables
trainable_variables
regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


Ekernel
Fbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 32]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 23}
?


Gkernel
Hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 64]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 28}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 30}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 31}
<
E0
F1
G2
H3"
trackable_list_wrapper
<
E0
F1
G2
H3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
 	variables
!trainable_variables
"regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


Ikernel
Jbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 36}
?


Kkernel
Lbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 128]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 41}
?


Mkernel
Nbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 128]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 46}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 48}}
J
I0
J1
K2
L3
M4
N5"
trackable_list_wrapper
J
I0
J1
K2
L3
M4
N5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
+	variables
,trainable_variables
-regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 50}}
?

?
activation

Okernel
Pbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 51}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 56}
?

?
activation

Qkernel
Rbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 57}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 61}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Skernel
Tbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 62}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 63}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 64, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ukernel
Vbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 66}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 67}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 68, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Wkernel
Xbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 70}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 71}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 72, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ykernel
Zbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 74}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 75}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 76, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

[kernel
\bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 11, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 78}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 79}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 80, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13"
trackable_list_wrapper
?
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
8	variables
9trainable_variables
:regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
@:> 2&simple_model2/conv_block/conv2d/kernel
2:0 2$simple_model2/conv_block/conv2d/bias
B:@  2(simple_model2/conv_block/conv2d_1/kernel
4:2 2&simple_model2/conv_block/conv2d_1/bias
D:B @2*simple_model2/conv_block_1/conv2d_2/kernel
6:4@2(simple_model2/conv_block_1/conv2d_2/bias
D:B@@2*simple_model2/conv_block_1/conv2d_3/kernel
6:4@2(simple_model2/conv_block_1/conv2d_3/bias
J:H@?2/simple_model2/larger_conv_block/conv2d_4/kernel
<::?2-simple_model2/larger_conv_block/conv2d_4/bias
K:I??2/simple_model2/larger_conv_block/conv2d_5/kernel
<::?2-simple_model2/larger_conv_block/conv2d_5/bias
K:I??2/simple_model2/larger_conv_block/conv2d_6/kernel
<::?2-simple_model2/larger_conv_block/conv2d_6/bias
@:>
??2,simple_model2/prediction_block2/dense/kernel
9:7?2*simple_model2/prediction_block2/dense/bias
B:@
??2.simple_model2/prediction_block2/dense_1/kernel
;:9?2,simple_model2/prediction_block2/dense_1/bias
A:?	?2.simple_model2/prediction_block2/dense_2/kernel
::82,simple_model2/prediction_block2/dense_2/bias
A:?	?2.simple_model2/prediction_block2/dense_3/kernel
::82,simple_model2/prediction_block2/dense_3/bias
A:?	?2.simple_model2/prediction_block2/dense_4/kernel
::82,simple_model2/prediction_block2/dense_4/bias
A:?	?2.simple_model2/prediction_block2/dense_5/kernel
::82,simple_model2/prediction_block2/dense_5/bias
A:?	?2.simple_model2/prediction_block2/dense_6/kernel
::82,simple_model2/prediction_block2/dense_6/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
y
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
g	variables
htrainable_variables
iregularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
k	variables
ltrainable_variables
mregularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
o	variables
ptrainable_variables
qregularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
s	variables
ttrainable_variables
uregularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
w	variables
xtrainable_variables
yregularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
{	variables
|trainable_variables
}regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
$0
%1
&2
'3
(4
)5
*6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 51}
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 57}
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
/0
01
12
23
34
45
56
67
78"
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 82}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "output_1_loss", "dtype": "float32", "config": {"name": "output_1_loss", "dtype": "float32"}, "shared_object_id": 83}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "output_2_loss", "dtype": "float32", "config": {"name": "output_2_loss", "dtype": "float32"}, "shared_object_id": 84}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "output_3_loss", "dtype": "float32", "config": {"name": "output_3_loss", "dtype": "float32"}, "shared_object_id": 85}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "output_4_loss", "dtype": "float32", "config": {"name": "output_4_loss", "dtype": "float32"}, "shared_object_id": 86}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "output_5_loss", "dtype": "float32", "config": {"name": "output_5_loss", "dtype": "float32"}, "shared_object_id": 87}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "output_1_accuracy", "dtype": "float32", "config": {"name": "output_1_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 0}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "output_2_accuracy", "dtype": "float32", "config": {"name": "output_2_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "output_3_accuracy", "dtype": "float32", "config": {"name": "output_3_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 2}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "output_4_accuracy", "dtype": "float32", "config": {"name": "output_4_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 3}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "output_5_accuracy", "dtype": "float32", "config": {"name": "output_5_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 4}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?non_trainable_variables
?	variables
?trainable_variables
?regularization_losses
?layer_metrics
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
E:C 2-Adam/simple_model2/conv_block/conv2d/kernel/m
7:5 2+Adam/simple_model2/conv_block/conv2d/bias/m
G:E  2/Adam/simple_model2/conv_block/conv2d_1/kernel/m
9:7 2-Adam/simple_model2/conv_block/conv2d_1/bias/m
I:G @21Adam/simple_model2/conv_block_1/conv2d_2/kernel/m
;:9@2/Adam/simple_model2/conv_block_1/conv2d_2/bias/m
I:G@@21Adam/simple_model2/conv_block_1/conv2d_3/kernel/m
;:9@2/Adam/simple_model2/conv_block_1/conv2d_3/bias/m
O:M@?26Adam/simple_model2/larger_conv_block/conv2d_4/kernel/m
A:??24Adam/simple_model2/larger_conv_block/conv2d_4/bias/m
P:N??26Adam/simple_model2/larger_conv_block/conv2d_5/kernel/m
A:??24Adam/simple_model2/larger_conv_block/conv2d_5/bias/m
P:N??26Adam/simple_model2/larger_conv_block/conv2d_6/kernel/m
A:??24Adam/simple_model2/larger_conv_block/conv2d_6/bias/m
E:C
??23Adam/simple_model2/prediction_block2/dense/kernel/m
>:<?21Adam/simple_model2/prediction_block2/dense/bias/m
G:E
??25Adam/simple_model2/prediction_block2/dense_1/kernel/m
@:>?23Adam/simple_model2/prediction_block2/dense_1/bias/m
F:D	?25Adam/simple_model2/prediction_block2/dense_2/kernel/m
?:=23Adam/simple_model2/prediction_block2/dense_2/bias/m
F:D	?25Adam/simple_model2/prediction_block2/dense_3/kernel/m
?:=23Adam/simple_model2/prediction_block2/dense_3/bias/m
F:D	?25Adam/simple_model2/prediction_block2/dense_4/kernel/m
?:=23Adam/simple_model2/prediction_block2/dense_4/bias/m
F:D	?25Adam/simple_model2/prediction_block2/dense_5/kernel/m
?:=23Adam/simple_model2/prediction_block2/dense_5/bias/m
F:D	?25Adam/simple_model2/prediction_block2/dense_6/kernel/m
?:=23Adam/simple_model2/prediction_block2/dense_6/bias/m
E:C 2-Adam/simple_model2/conv_block/conv2d/kernel/v
7:5 2+Adam/simple_model2/conv_block/conv2d/bias/v
G:E  2/Adam/simple_model2/conv_block/conv2d_1/kernel/v
9:7 2-Adam/simple_model2/conv_block/conv2d_1/bias/v
I:G @21Adam/simple_model2/conv_block_1/conv2d_2/kernel/v
;:9@2/Adam/simple_model2/conv_block_1/conv2d_2/bias/v
I:G@@21Adam/simple_model2/conv_block_1/conv2d_3/kernel/v
;:9@2/Adam/simple_model2/conv_block_1/conv2d_3/bias/v
O:M@?26Adam/simple_model2/larger_conv_block/conv2d_4/kernel/v
A:??24Adam/simple_model2/larger_conv_block/conv2d_4/bias/v
P:N??26Adam/simple_model2/larger_conv_block/conv2d_5/kernel/v
A:??24Adam/simple_model2/larger_conv_block/conv2d_5/bias/v
P:N??26Adam/simple_model2/larger_conv_block/conv2d_6/kernel/v
A:??24Adam/simple_model2/larger_conv_block/conv2d_6/bias/v
E:C
??23Adam/simple_model2/prediction_block2/dense/kernel/v
>:<?21Adam/simple_model2/prediction_block2/dense/bias/v
G:E
??25Adam/simple_model2/prediction_block2/dense_1/kernel/v
@:>?23Adam/simple_model2/prediction_block2/dense_1/bias/v
F:D	?25Adam/simple_model2/prediction_block2/dense_2/kernel/v
?:=23Adam/simple_model2/prediction_block2/dense_2/bias/v
F:D	?25Adam/simple_model2/prediction_block2/dense_3/kernel/v
?:=23Adam/simple_model2/prediction_block2/dense_3/bias/v
F:D	?25Adam/simple_model2/prediction_block2/dense_4/kernel/v
?:=23Adam/simple_model2/prediction_block2/dense_4/bias/v
F:D	?25Adam/simple_model2/prediction_block2/dense_5/kernel/v
?:=23Adam/simple_model2/prediction_block2/dense_5/bias/v
F:D	?25Adam/simple_model2/prediction_block2/dense_6/kernel/v
?:=23Adam/simple_model2/prediction_block2/dense_6/bias/v
?2?
.__inference_simple_model2_layer_call_fn_126034
.__inference_simple_model2_layer_call_fn_126905
.__inference_simple_model2_layer_call_fn_126974
.__inference_simple_model2_layer_call_fn_126613?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
!__inference__wrapped_model_125718?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
I__inference_simple_model2_layer_call_and_return_conditional_losses_127089
I__inference_simple_model2_layer_call_and_return_conditional_losses_127232
I__inference_simple_model2_layer_call_and_return_conditional_losses_126686
I__inference_simple_model2_layer_call_and_return_conditional_losses_126759?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_127237
(__inference_dropout_layer_call_fn_127242?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_127247
C__inference_dropout_layer_call_and_return_conditional_losses_127259?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_conv_block_layer_call_fn_127272
+__inference_conv_block_layer_call_fn_127285?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv_block_layer_call_and_return_conditional_losses_127305
F__inference_conv_block_layer_call_and_return_conditional_losses_127332?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv_block_1_layer_call_fn_127345
-__inference_conv_block_1_layer_call_fn_127358?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv_block_1_layer_call_and_return_conditional_losses_127378
H__inference_conv_block_1_layer_call_and_return_conditional_losses_127405?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_larger_conv_block_layer_call_fn_127422?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_127448?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_prediction_block2_layer_call_fn_127489
2__inference_prediction_block2_layer_call_fn_127530?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_prediction_block2_layer_call_and_return_conditional_losses_127590
M__inference_prediction_block2_layer_call_and_return_conditional_losses_127657?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_126836input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_layer_call_fn_125730?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_125724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_1_layer_call_fn_125742?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_125736?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_2_layer_call_fn_125754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_125748?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_125718?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\8?5
.?+
)?&
input_1?????????@@
? "???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
.
output_3"?
output_3?????????
.
output_4"?
output_4?????????
.
output_5"?
output_5??????????
H__inference_conv_block_1_layer_call_and_return_conditional_losses_127378rEFGH;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0?????????@
? ?
H__inference_conv_block_1_layer_call_and_return_conditional_losses_127405rEFGH;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0?????????@
? ?
-__inference_conv_block_1_layer_call_fn_127345eEFGH;?8
1?.
(?%
inputs????????? 
p 
? " ??????????@?
-__inference_conv_block_1_layer_call_fn_127358eEFGH;?8
1?.
(?%
inputs????????? 
p
? " ??????????@?
F__inference_conv_block_layer_call_and_return_conditional_losses_127305rABCD;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0????????? 
? ?
F__inference_conv_block_layer_call_and_return_conditional_losses_127332rABCD;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0????????? 
? ?
+__inference_conv_block_layer_call_fn_127272eABCD;?8
1?.
(?%
inputs?????????@@
p 
? " ?????????? ?
+__inference_conv_block_layer_call_fn_127285eABCD;?8
1?.
(?%
inputs?????????@@
p
? " ?????????? ?
C__inference_dropout_layer_call_and_return_conditional_losses_127247l;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_127259l;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
(__inference_dropout_layer_call_fn_127237_;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
(__inference_dropout_layer_call_fn_127242_;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
M__inference_larger_conv_block_layer_call_and_return_conditional_losses_127448qIJKLMN7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
2__inference_larger_conv_block_layer_call_fn_127422dIJKLMN7?4
-?*
(?%
inputs?????????@
? "!????????????
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_125736?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_1_layer_call_fn_125742?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_125748?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_2_layer_call_fn_125754?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_125724?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_125730?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_prediction_block2_layer_call_and_return_conditional_losses_127590?OPQRSTUVWXYZ[\<?9
2?/
)?&
inputs??????????
p 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
? ?
M__inference_prediction_block2_layer_call_and_return_conditional_losses_127657?OPQRSTUVWXYZ[\<?9
2?/
)?&
inputs??????????
p
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
? ?
2__inference_prediction_block2_layer_call_fn_127489?OPQRSTUVWXYZ[\<?9
2?/
)?&
inputs??????????
p 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4??????????
2__inference_prediction_block2_layer_call_fn_127530?OPQRSTUVWXYZ[\<?9
2?/
)?&
inputs??????????
p
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4??????????
$__inference_signature_wrapper_126836?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\C?@
? 
9?6
4
input_1)?&
input_1?????????@@"???
.
output_1"?
output_1?????????
.
output_2"?
output_2?????????
.
output_3"?
output_3?????????
.
output_4"?
output_4?????????
.
output_5"?
output_5??????????
I__inference_simple_model2_layer_call_and_return_conditional_losses_126686?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\<?9
2?/
)?&
input_1?????????@@
p 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
? ?
I__inference_simple_model2_layer_call_and_return_conditional_losses_126759?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\<?9
2?/
)?&
input_1?????????@@
p
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
? ?
I__inference_simple_model2_layer_call_and_return_conditional_losses_127089?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\;?8
1?.
(?%
inputs?????????@@
p 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
? ?
I__inference_simple_model2_layer_call_and_return_conditional_losses_127232?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\;?8
1?.
(?%
inputs?????????@@
p
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
? ?
.__inference_simple_model2_layer_call_fn_126034?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\<?9
2?/
)?&
input_1?????????@@
p 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4??????????
.__inference_simple_model2_layer_call_fn_126613?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\<?9
2?/
)?&
input_1?????????@@
p
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4??????????
.__inference_simple_model2_layer_call_fn_126905?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\;?8
1?.
(?%
inputs?????????@@
p 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4??????????
.__inference_simple_model2_layer_call_fn_126974?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\;?8
1?.
(?%
inputs?????????@@
p
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????