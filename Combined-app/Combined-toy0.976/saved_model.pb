��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8��
�
sequential_1_1/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namesequential_1_1/dense_2/kernel
�
1sequential_1_1/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential_1_1/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
sequential_1_1/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namesequential_1_1/dense_2/bias
�
/sequential_1_1/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential_1_1/dense_2/bias*
_output_shapes	
:�*
dtype0
�
sequential_1_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�I*.
shared_namesequential_1_1/dense_3/kernel
�
1sequential_1_1/dense_3/kernel/Read/ReadVariableOpReadVariableOpsequential_1_1/dense_3/kernel*
_output_shapes
:	�I*
dtype0
�
sequential_1_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:I*,
shared_namesequential_1_1/dense_3/bias
�
/sequential_1_1/dense_3/bias/Read/ReadVariableOpReadVariableOpsequential_1_1/dense_3/bias*
_output_shapes
:I*
dtype0
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
�
$Adam/sequential_1_1/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/sequential_1_1/dense_2/kernel/m
�
8Adam/sequential_1_1/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/sequential_1_1/dense_2/kernel/m* 
_output_shapes
:
��*
dtype0
�
"Adam/sequential_1_1/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/sequential_1_1/dense_2/bias/m
�
6Adam/sequential_1_1/dense_2/bias/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_1_1/dense_2/bias/m*
_output_shapes	
:�*
dtype0
�
$Adam/sequential_1_1/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�I*5
shared_name&$Adam/sequential_1_1/dense_3/kernel/m
�
8Adam/sequential_1_1/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/sequential_1_1/dense_3/kernel/m*
_output_shapes
:	�I*
dtype0
�
"Adam/sequential_1_1/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:I*3
shared_name$"Adam/sequential_1_1/dense_3/bias/m
�
6Adam/sequential_1_1/dense_3/bias/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_1_1/dense_3/bias/m*
_output_shapes
:I*
dtype0
�
$Adam/sequential_1_1/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/sequential_1_1/dense_2/kernel/v
�
8Adam/sequential_1_1/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/sequential_1_1/dense_2/kernel/v* 
_output_shapes
:
��*
dtype0
�
"Adam/sequential_1_1/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/sequential_1_1/dense_2/bias/v
�
6Adam/sequential_1_1/dense_2/bias/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_1_1/dense_2/bias/v*
_output_shapes	
:�*
dtype0
�
$Adam/sequential_1_1/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�I*5
shared_name&$Adam/sequential_1_1/dense_3/kernel/v
�
8Adam/sequential_1_1/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/sequential_1_1/dense_3/kernel/v*
_output_shapes
:	�I*
dtype0
�
"Adam/sequential_1_1/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:I*3
shared_name$"Adam/sequential_1_1/dense_3/bias/v
�
6Adam/sequential_1_1/dense_3/bias/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_1_1/dense_3/bias/v*
_output_shapes
:I*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�
iter

beta_1

beta_2
	decay
learning_rate	m2
m3m4m5	v6
v7v8v9

	0

1
2
3
 

	0

1
2
3
�
metrics
trainable_variables

layers
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
 
\Z
VARIABLE_VALUEsequential_1_1/dense_2/kernel)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_1_1/dense_2/bias'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
�
metrics
trainable_variables

layers
 non_trainable_variables
	variables
regularization_losses
!layer_regularization_losses
\Z
VARIABLE_VALUEsequential_1_1/dense_3/kernel)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_1_1/dense_3/bias'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
"metrics
trainable_variables

#layers
$non_trainable_variables
	variables
regularization_losses
%layer_regularization_losses
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

&0

0
1
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
x
	'total
	(count
)
_fn_kwargs
*trainable_variables
+	variables
,regularization_losses
-	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

'0
(1
 
�
.metrics
*trainable_variables

/layers
0non_trainable_variables
+	variables
,regularization_losses
1layer_regularization_losses
 
 

'0
(1
 
}
VARIABLE_VALUE$Adam/sequential_1_1/dense_2/kernel/mElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_1_1/dense_2/bias/mClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_1_1/dense_3/kernel/mElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_1_1/dense_3/bias/mClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_1_1/dense_2/kernel/vElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_1_1/dense_2/bias/vClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_1_1/dense_3/kernel/vElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_1_1/dense_3/bias/vClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential_1_1/dense_2/kernelsequential_1_1/dense_2/biassequential_1_1/dense_3/kernelsequential_1_1/dense_3/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*.
f)R'
%__inference_signature_wrapper_2097235
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1sequential_1_1/dense_2/kernel/Read/ReadVariableOp/sequential_1_1/dense_2/bias/Read/ReadVariableOp1sequential_1_1/dense_3/kernel/Read/ReadVariableOp/sequential_1_1/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp8Adam/sequential_1_1/dense_2/kernel/m/Read/ReadVariableOp6Adam/sequential_1_1/dense_2/bias/m/Read/ReadVariableOp8Adam/sequential_1_1/dense_3/kernel/m/Read/ReadVariableOp6Adam/sequential_1_1/dense_3/bias/m/Read/ReadVariableOp8Adam/sequential_1_1/dense_2/kernel/v/Read/ReadVariableOp6Adam/sequential_1_1/dense_2/bias/v/Read/ReadVariableOp8Adam/sequential_1_1/dense_3/kernel/v/Read/ReadVariableOp6Adam/sequential_1_1/dense_3/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__traced_save_2097406
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_1_1/dense_2/kernelsequential_1_1/dense_2/biassequential_1_1/dense_3/kernelsequential_1_1/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount$Adam/sequential_1_1/dense_2/kernel/m"Adam/sequential_1_1/dense_2/bias/m$Adam/sequential_1_1/dense_3/kernel/m"Adam/sequential_1_1/dense_3/bias/m$Adam/sequential_1_1/dense_2/kernel/v"Adam/sequential_1_1/dense_2/bias/v$Adam/sequential_1_1/dense_3/kernel/v"Adam/sequential_1_1/dense_3/bias/v*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__traced_restore_2097475��
�
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097271

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_2/Relu�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�I*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������I2
dense_3/Softmax�
IdentityIdentitydense_3/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
D__inference_dense_3_layer_call_and_return_conditional_losses_2097155

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�I*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:I*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������I2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�S
�
#__inference__traced_restore_2097475
file_prefix2
.assignvariableop_sequential_1_1_dense_2_kernel2
.assignvariableop_1_sequential_1_1_dense_2_bias4
0assignvariableop_2_sequential_1_1_dense_3_kernel2
.assignvariableop_3_sequential_1_1_dense_3_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count<
8assignvariableop_11_adam_sequential_1_1_dense_2_kernel_m:
6assignvariableop_12_adam_sequential_1_1_dense_2_bias_m<
8assignvariableop_13_adam_sequential_1_1_dense_3_kernel_m:
6assignvariableop_14_adam_sequential_1_1_dense_3_bias_m<
8assignvariableop_15_adam_sequential_1_1_dense_2_kernel_v:
6assignvariableop_16_adam_sequential_1_1_dense_2_bias_v<
8assignvariableop_17_adam_sequential_1_1_dense_3_kernel_v:
6assignvariableop_18_adam_sequential_1_1_dense_3_bias_v
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp.assignvariableop_sequential_1_1_dense_2_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_sequential_1_1_dense_2_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_sequential_1_1_dense_3_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_sequential_1_1_dense_3_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp8assignvariableop_11_adam_sequential_1_1_dense_2_kernel_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp6assignvariableop_12_adam_sequential_1_1_dense_2_bias_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_sequential_1_1_dense_3_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_sequential_1_1_dense_3_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp8assignvariableop_15_adam_sequential_1_1_dense_2_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_sequential_1_1_dense_2_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_sequential_1_1_dense_3_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_sequential_1_1_dense_3_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19�
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�	
�
D__inference_dense_3_layer_call_and_return_conditional_losses_2097318

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�I*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:I*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������I2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
.__inference_sequential_1_layer_call_fn_2097280

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_20971912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097210

inputs*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20971322!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_20971552!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
.__inference_sequential_1_layer_call_fn_2097198
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_20971912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
.__inference_sequential_1_layer_call_fn_2097289

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_20972102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_2097235
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__wrapped_model_20971172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097253

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_2/Relu�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	�I*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������I2
dense_3/Softmax�
IdentityIdentitydense_3/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097191

inputs*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20971322!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_20971552!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097168
input_1*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20971322!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_20971552!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
.__inference_sequential_1_layer_call_fn_2097217
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_20972102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
"__inference__wrapped_model_2097117
input_17
3sequential_1_dense_2_matmul_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource
identity��+sequential_1/dense_2/BiasAdd/ReadVariableOp�*sequential_1/dense_2/MatMul/ReadVariableOp�+sequential_1/dense_3/BiasAdd/ReadVariableOp�*sequential_1/dense_3/MatMul/ReadVariableOp�
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02,
*sequential_1/dense_2/MatMul/ReadVariableOp�
sequential_1/dense_2/MatMulMatMulinput_12sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_1/dense_2/MatMul�
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp�
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_1/dense_2/BiasAdd�
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_1/dense_2/Relu�
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�I*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp�
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
sequential_1/dense_3/MatMul�
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:I*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp�
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������I2
sequential_1/dense_3/BiasAdd�
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������I2
sequential_1/dense_3/Softmax�
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
�
�
)__inference_dense_2_layer_call_fn_2097307

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20971322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�1
�	
 __inference__traced_save_2097406
file_prefix<
8savev2_sequential_1_1_dense_2_kernel_read_readvariableop:
6savev2_sequential_1_1_dense_2_bias_read_readvariableop<
8savev2_sequential_1_1_dense_3_kernel_read_readvariableop:
6savev2_sequential_1_1_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopC
?savev2_adam_sequential_1_1_dense_2_kernel_m_read_readvariableopA
=savev2_adam_sequential_1_1_dense_2_bias_m_read_readvariableopC
?savev2_adam_sequential_1_1_dense_3_kernel_m_read_readvariableopA
=savev2_adam_sequential_1_1_dense_3_bias_m_read_readvariableopC
?savev2_adam_sequential_1_1_dense_2_kernel_v_read_readvariableopA
=savev2_adam_sequential_1_1_dense_2_bias_v_read_readvariableopC
?savev2_adam_sequential_1_1_dense_3_kernel_v_read_readvariableopA
=savev2_adam_sequential_1_1_dense_3_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9132c98da79b4f3eaf4d5e87ad222ed7/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_sequential_1_1_dense_2_kernel_read_readvariableop6savev2_sequential_1_1_dense_2_bias_read_readvariableop8savev2_sequential_1_1_dense_3_kernel_read_readvariableop6savev2_sequential_1_1_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop?savev2_adam_sequential_1_1_dense_2_kernel_m_read_readvariableop=savev2_adam_sequential_1_1_dense_2_bias_m_read_readvariableop?savev2_adam_sequential_1_1_dense_3_kernel_m_read_readvariableop=savev2_adam_sequential_1_1_dense_3_bias_m_read_readvariableop?savev2_adam_sequential_1_1_dense_2_kernel_v_read_readvariableop=savev2_adam_sequential_1_1_dense_2_bias_v_read_readvariableop?savev2_adam_sequential_1_1_dense_3_kernel_v_read_readvariableop=savev2_adam_sequential_1_1_dense_3_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
~: :
��:�:	�I:I: : : : : : : :
��:�:	�I:I:
��:�:	�I:I: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�	
�
D__inference_dense_2_layer_call_and_return_conditional_losses_2097300

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097178
input_1*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2*
&dense_3_statefulpartitionedcall_args_1*
&dense_3_statefulpartitionedcall_args_2
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_20971322!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0&dense_3_statefulpartitionedcall_args_1&dense_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_20971552!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�	
�
D__inference_dense_2_layer_call_and_return_conditional_losses_2097132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_dense_3_layer_call_fn_2097325

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������I*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_20971552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������I2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_11
serving_default_input_1:0����������<
output_10
StatefulPartitionedCall:0���������Itensorflow/serving/predict:�h
�
layer-0
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*:&call_and_return_all_conditional_losses
;__call__
<_default_save_signature"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 73, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 266]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 266}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 73, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 266]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 266}}}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 73, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�
iter

beta_1

beta_2
	decay
learning_rate	m2
m3m4m5	v6
v7v8v9"
	optimizer
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
�
metrics
trainable_variables

layers
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
;__call__
<_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
Aserving_default"
signature_map
1:/
��2sequential_1_1/dense_2/kernel
*:(�2sequential_1_1/dense_2/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
metrics
trainable_variables

layers
 non_trainable_variables
	variables
regularization_losses
!layer_regularization_losses
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
0:.	�I2sequential_1_1/dense_3/kernel
):'I2sequential_1_1/dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
"metrics
trainable_variables

#layers
$non_trainable_variables
	variables
regularization_losses
%layer_regularization_losses
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
'
&0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
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
�
	'total
	(count
)
_fn_kwargs
*trainable_variables
+	variables
,regularization_losses
-	keras_api
*B&call_and_return_all_conditional_losses
C__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
.metrics
*trainable_variables

/layers
0non_trainable_variables
+	variables
,regularization_losses
1layer_regularization_losses
C__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
6:4
��2$Adam/sequential_1_1/dense_2/kernel/m
/:-�2"Adam/sequential_1_1/dense_2/bias/m
5:3	�I2$Adam/sequential_1_1/dense_3/kernel/m
.:,I2"Adam/sequential_1_1/dense_3/bias/m
6:4
��2$Adam/sequential_1_1/dense_2/kernel/v
/:-�2"Adam/sequential_1_1/dense_2/bias/v
5:3	�I2$Adam/sequential_1_1/dense_3/kernel/v
.:,I2"Adam/sequential_1_1/dense_3/bias/v
�2�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097271
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097253
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097178
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097168�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_sequential_1_layer_call_fn_2097217
.__inference_sequential_1_layer_call_fn_2097280
.__inference_sequential_1_layer_call_fn_2097198
.__inference_sequential_1_layer_call_fn_2097289�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_2097117�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *'�$
"�
input_1����������
�2�
D__inference_dense_2_layer_call_and_return_conditional_losses_2097300�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_2_layer_call_fn_2097307�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_3_layer_call_and_return_conditional_losses_2097318�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_3_layer_call_fn_2097325�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
4B2
%__inference_signature_wrapper_2097235input_1
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
"__inference__wrapped_model_2097117n	
1�.
'�$
"�
input_1����������
� "3�0
.
output_1"�
output_1���������I�
D__inference_dense_2_layer_call_and_return_conditional_losses_2097300^	
0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_2_layer_call_fn_2097307Q	
0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_3_layer_call_and_return_conditional_losses_2097318]0�-
&�#
!�
inputs����������
� "%�"
�
0���������I
� }
)__inference_dense_3_layer_call_fn_2097325P0�-
&�#
!�
inputs����������
� "����������I�
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097168h	
9�6
/�,
"�
input_1����������
p

 
� "%�"
�
0���������I
� �
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097178h	
9�6
/�,
"�
input_1����������
p 

 
� "%�"
�
0���������I
� �
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097253g	
8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������I
� �
I__inference_sequential_1_layer_call_and_return_conditional_losses_2097271g	
8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������I
� �
.__inference_sequential_1_layer_call_fn_2097198[	
9�6
/�,
"�
input_1����������
p

 
� "����������I�
.__inference_sequential_1_layer_call_fn_2097217[	
9�6
/�,
"�
input_1����������
p 

 
� "����������I�
.__inference_sequential_1_layer_call_fn_2097280Z	
8�5
.�+
!�
inputs����������
p

 
� "����������I�
.__inference_sequential_1_layer_call_fn_2097289Z	
8�5
.�+
!�
inputs����������
p 

 
� "����������I�
%__inference_signature_wrapper_2097235y	
<�9
� 
2�/
-
input_1"�
input_1����������"3�0
.
output_1"�
output_1���������I